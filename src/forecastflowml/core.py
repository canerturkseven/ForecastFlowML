import pickle
import re
import datetime
import sklearn
import pyspark
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from forecastflowml.model_selection import (
    _cross_val_forecast,
    _score_func,
    _TimeBasedSplit,
)
from forecastflowml.utils import _check_input_type, _check_spark, _check_fitted
from typing import List, Optional, Union, Dict


class ForecastFlowML:
    """Create forecaster Instance

    Parameters
    ----------
    id_col
        Time series identifer column.
    group_col
        Column to partition the dataframe.
    date_col
        Date column.
    target_col
        Target column.
    date_frequency
        Date frequency of the dataframe.
    model_horizon
        Forecast horizon for a single model.
    max_forecast_horizon
        Maximum horizon to generate the forecast. Needs to be multiple of
        ``model_horizon``.
    model
        Regressor compatible with ``scikit-learn`` API.
    categorical_cols
        List of columns to treat as categorical.
    use_lag_range
        Extra lag range to use in addition to allowed lag values.
    """

    def __init__(
        self,
        id_col: str,
        group_col: str,
        date_col: str,
        target_col: str,
        date_frequency: str,
        max_forecast_horizon: int,
        model_horizon: int,
        model: sklearn.base.BaseEstimator,
        categorical_cols: Optional[List[str]] = None,
        use_lag_range: int = 0,
    ) -> None:
        self.id_col = id_col
        self.group_col = group_col
        self.date_col = date_col
        self.target_col = target_col
        self.categorical_cols = categorical_cols
        self.date_frequency = date_frequency
        self.model = model
        self.max_forecast_horizon = max_forecast_horizon
        self.model_horizon = model_horizon
        self.use_lag_range = use_lag_range
        self.n_horizon = max_forecast_horizon // model_horizon

    @property
    def model_(self) -> pd.DataFrame:
        """Trained models in pickled format"""
        return self._model_

    @model_.setter
    def model_(self, value: pd.DataFrame) -> None:
        """Set models attribute after training

        Parameters
        ----------
        value
            pandas DataFrame containing trained models
        """
        self._model_ = value

    def _filter_horizon(self, df, forecast_horizon):
        dates = df[self.date_col].sort_values().unique()
        forecast_dates = dates[[fh - 1 for fh in forecast_horizon]]
        return df[df[self.date_col].isin(forecast_dates)]

    def _filter_features(self, df, forecast_horizon):
        min_lag = max(forecast_horizon)
        lag_range = self.use_lag_range
        feature_cols = [
            col
            for col in df.select_dtypes(["number", "category"]).columns
            if col not in [self.id_col, self.group_col, self.date_col, self.target_col]
        ]
        lag_cols = [
            col
            for col in feature_cols
            if re.findall("(^|_)lag(_|$)", col, re.IGNORECASE)
        ]
        keep_lags_str = "|".join(map(str, range(min_lag, min_lag + lag_range + 1)))
        keep_lags = [
            col
            for col in lag_cols
            if re.findall(
                f"^lag_({keep_lags_str})$|(^|_)lag_{min_lag}(_|$)", col, re.IGNORECASE
            )
        ]
        features = list(set(feature_cols) - set(lag_cols)) + keep_lags
        return features

    def _forecast_horizon(self, i):
        model_horizon = self.model_horizon
        return list(range(i * model_horizon + 1, (i + 1) * model_horizon + 1))

    def _convert_categorical(self, df):
        categorical_cols = self.categorical_cols
        if categorical_cols is not None:
            df[categorical_cols] = df[categorical_cols].astype("category")
        return df

    def _serialize(self, df):
        group_col = self.group_col

        @F.pandas_udf(
            "group string, data binary",
            functionType=F.PandasUDFType.GROUPED_MAP,
        )
        def _serialize_udf(df):
            return pd.DataFrame(
                [{"group": df[group_col].iloc[0], "data": pickle.dumps(df)}]
            )

        return df.groupby(group_col).apply(_serialize_udf)

    def get_feature_importance(
        self,
        df_model: Optional[pyspark.sql.DataFrame] = None,
    ) -> pd.DataFrame:
        """The feature importances.

        Parameters
        ----------
        df_model
            pyspark DataFrame that contains the trained models. Only needs to be
            supplied if ``local_result`` is set to ``False`` during training.

        Returns
        -------
            DataFrame that includes the feature importances.
        """

        def _feature_importance_udf(df):
            group = df["group"].iloc[0]

            importance_list = []
            for i in range(len(df["model"].iloc[0])):
                model = pickle.loads(df["model"].iloc[0][i])
                forecast_horizon = df["forecast_horizon"].iloc[0][i]

                importance = pd.DataFrame(
                    zip(
                        [forecast_horizon] * model.n_features_,
                        model.feature_name_,
                        model.feature_importances_,
                    ),
                    columns=["forecast_horizon", "feature", "importance"],
                )
                importance_list.append(importance)

            df_importance = pd.concat(importance_list)
            df_importance.insert(0, "group", group)

            return df_importance

        if df_model is not None:
            pandas_udf = F.pandas_udf(
                _feature_importance_udf,
                "group:string, forecast_horizon:array<int>, "
                "feature:string, importance:float",
                functionType=F.PandasUDFType.GROUPED_MAP,
            )
            return df_model.groupby("group").apply(pandas_udf).toPandas()
        else:
            return (
                self.model_.groupby("group", group_keys=False)
                .apply(_feature_importance_udf)
                .reset_index(drop=True)
            )

    def train(
        self,
        df: Union[pd.DataFrame, pyspark.sql.DataFrame],
        spark: Optional[SparkSession] = None,
        local_result: bool = False,
    ) -> Union[None, pd.DataFrame]:
        """Train models

        Parameters
        ----------
        df
            Dataset to fit.
        spark
            Spark session instance. Only provide when ``df`` is a pandas DataFrame.
        local_result
            Whether to store trained models as attribute. Only provide ``True``
            in case of the trained models are not expected to overload the driver node.

        Returns
        -------
            None if ``df`` is pandas DataFrame or ``local_result=True``. Otherwise, pyspark DataFrame that includes the trained models.
        """

        group_col = self.group_col
        target_col = self.target_col
        model = self.model
        input_type = _check_input_type(df)
        _check_spark(self, input_type, spark)

        @F.pandas_udf(
            "group:string, forecast_horizon:array<array<int>>, model:array<binary>,"
            "start_time:string, end_time:string, elapsed_seconds:float",
            functionType=F.PandasUDFType.GROUPED_MAP,
        )
        def _train_udf(df):
            start = datetime.datetime.now()

            df = self._convert_categorical(df)
            group = df[group_col].iloc[0]
            group_model = model[group] if isinstance(model, dict) else model

            model_list = []
            forecast_horizon_list = []
            for i in range(self.n_horizon):
                forecast_horizon = self._forecast_horizon(i)
                features = self._filter_features(df, forecast_horizon)

                X = df[features]
                y = df[target_col]

                group_model.fit(X, y)

                forecast_horizon_list.append(forecast_horizon)
                model_list.append(pickle.dumps(group_model))

            end = datetime.datetime.now()
            elapsed = end - start
            seconds = round(elapsed.total_seconds(), 1)

            return pd.DataFrame(
                [
                    {
                        "group": group,
                        "forecast_horizon": forecast_horizon_list,
                        "model": model_list,
                        "start_time": start.strftime("%d-%b-%Y (%H:%M:%S)"),
                        "end_time": end.strftime("%d-%b-%Y (%H:%M:%S)"),
                        "elapsed_seconds": seconds,
                    },
                ]
            )

        df = spark.createDataFrame(df) if input_type == "df_pandas" else df
        model_ = (
            df.withColumn("date", F.to_timestamp("date"))
            .groupby(group_col)
            .apply(_train_udf)
        )
        if (input_type == "df_pandas") | (local_result):
            self.model_ = model_.toPandas()
        else:
            return model_

    def cross_validate(
        self,
        df,
        n_cv_splits: int = 3,
        max_train_size: Optional[int] = None,
        cv_step_length: Optional[int] = None,
        refit: bool = True,
        spark: Optional[SparkSession] = None,
    ) -> Union[pd.DataFrame, pyspark.sql.DataFrame]:
        """Time series cross validation predictions

        Parameters
        ----------
        df
            Dataset to fit.
        n_cv_splits
            Number of cross validation folds.
        max_train_size
            Number of max periods to use as training set.
        cv_step_length
            Number of periods to put between each cv folds.
        refit
            Whether to refit model for each training dataset.
        spark
            Spark session instance. Only provide when ``df`` is a pandas DataFrame.

        Returns
        -------
            DataFrame that contains target and predictions over cross validation folds.
        """

        id_col = self.id_col
        target_col = self.target_col
        date_col = self.date_col
        date_frequency = self.date_frequency
        max_forecast_horizon = self.max_forecast_horizon
        group_col = self.group_col
        model = self.model
        cv_step_length = (
            max_forecast_horizon if cv_step_length is None else cv_step_length
        )
        input_type = _check_input_type(df)
        _check_spark(self, input_type, spark)

        @F.pandas_udf(
            "group string, id string, date date, cv string,"
            "target float, forecast float",
            functionType=F.PandasUDFType.GROUPED_MAP,
        )
        def _cross_validate_udf(df):
            df = self._convert_categorical(df)
            group = df[group_col].iloc[0]
            group_model = model[group] if isinstance(model, dict) else model

            cv_forecast_list = []
            for i in range(self.n_horizon):
                forecast_horizon = self._forecast_horizon(i)
                features = self._filter_features(df, forecast_horizon)

                cv = _TimeBasedSplit(
                    date_col=date_col,
                    date_frequency=date_frequency,
                    n_splits=int(n_cv_splits),
                    forecast_horizon=list(forecast_horizon),
                    step_length=int(cv_step_length),
                    max_train_size=max_train_size,
                    end_offset=int(max_forecast_horizon - int(max(forecast_horizon))),
                ).split(df)

                cv_forecast = _cross_val_forecast(
                    model=group_model,
                    df=df,
                    id_col=id_col,
                    feature_cols=features,
                    date_col=date_col,
                    target_col=target_col,
                    cv=cv,
                    refit=refit,
                )
                cv_forecast_list.append(cv_forecast)

            cv_forecast = pd.concat(cv_forecast_list).reset_index(drop=True)
            cv_forecast.insert(0, "group", group)

            return cv_forecast

        df = spark.createDataFrame(df) if input_type == "df_pandas" else df
        cv_result = (
            df.withColumn("date", F.to_timestamp("date"))
            .groupby(group_col)
            .apply(_cross_validate_udf)
        )
        if input_type == "df_pandas":
            return cv_result.toPandas()
        else:
            return cv_result

    def grid_search(
        self,
        df: Union[pd.DataFrame, pyspark.sql.DataFrame],
        param_grid: Dict[str, List[Union[str, float, int]]],
        n_cv_splits: int = 3,
        cv_step_length: Optional[int] = None,
        scoring_metric: str = "neg_mean_squared_error",
        refit: bool = True,
        spark: Optional[SparkSession] = None,
    ) -> pd.DataFrame:
        """Grid search with time series cross validation.

        Parameters
        ----------
        df
            Dataset to fit.
        param_grid
            Dictionary with parameters as keys and lists of parameter settings
            to try as values.
        n_cv_splits
            Number of cross validation folds.
        cv_step_length
            Number of periods to put between each cv folds.
        scoring_metric
            ``scikit-learn`` scoring metric.
            See list of available metrics: https://scikit-learn.org/stable/modules/model_evaluation.html.
        refit
            Whether to refit model for each training dataset.
        spark
            Spark session instance. Only provide when ``df`` is a pandas DataFrame.

        Returns
        -------
            DataFrame that includes score per parameter combination.
        """
        group_col = self.group_col
        id_col = self.id_col
        model = self.model
        target_col = self.target_col
        date_col = self.date_col
        date_frequency = self.date_frequency
        max_forecast_horizon = self.max_forecast_horizon
        cv_step_length = (
            max_forecast_horizon if cv_step_length is None else cv_step_length
        )
        input_type = _check_input_type(df)
        _check_spark(self, input_type, spark)

        @F.pandas_udf(
            (
                "group string, score float, "
                + ", ".join(
                    [
                        f"{key} {type(value[0]).__name__}"
                        for key, value in param_grid.items()
                    ]
                )
            ),
            functionType=F.PandasUDFType.GROUPED_MAP,
        )
        def _grid_search_udf(df):
            df = self._convert_categorical(df)
            group = df[group_col].iloc[0]
            hyperparams = df[list(param_grid.keys())].iloc[0].to_dict()
            group_model = model.set_params(**hyperparams)

            cv_forecast_list = []
            for i in range(self.n_horizon):
                forecast_horizon = self._forecast_horizon(i)
                features = self._filter_features(df, forecast_horizon)

                cv = _TimeBasedSplit(
                    date_col=date_col,
                    date_frequency=date_frequency,
                    n_splits=int(n_cv_splits),
                    forecast_horizon=list(forecast_horizon),
                    step_length=int(cv_step_length),
                    end_offset=int(max_forecast_horizon - int(max(forecast_horizon))),
                ).split(df)

                cv_forecast = _cross_val_forecast(
                    model=group_model,
                    df=df,
                    id_col=id_col,
                    feature_cols=features,
                    date_col=date_col,
                    target_col=target_col,
                    cv=cv,
                    refit=refit,
                )
                cv_forecast_list.append(cv_forecast)

            cv_forecast = pd.concat(cv_forecast_list)
            score = (
                cv_forecast.groupby("cv")
                .apply(
                    lambda x: _score_func(x["target"], x["forecast"], scoring_metric)
                )
                .mean()
            )

            return pd.DataFrame(
                [
                    {
                        **{
                            "group": group,
                            "score": score,
                        },
                        **hyperparams,
                    }
                ]
            )

        df = spark.createDataFrame(df) if input_type == "df_pandas" else df
        df = df.withColumn("date", F.to_timestamp("date"))

        for key in param_grid.keys():
            values = param_grid[key]
            column = F.explode(F.array([F.lit(v) for v in values]))
            df = df.withColumn(key, column)

        return (
            df.groupby([group_col, *param_grid.keys()])
            .apply(_grid_search_udf)
            .toPandas()
            .sort_values(by=["group", "score"], ascending=False)
            .reset_index(drop=True)
        )

    def _predict_grid(self, df, trained_models):
        df = self._serialize(df)
        df = df.join(
            trained_models.select("group", "forecast_horizon", "model"),
            on="group",
            how="left",
        )
        return df

    def predict(
        self,
        df: pd.DataFrame,
        trained_models=None,
        spark=None,
    ) -> Union[pd.DataFrame, pyspark.sql.DataFrame]:
        """Make predictions

        Parameters
        ----------
        df
            Dataset to perform predictions on.
        trained_models
            pyspark DataFrame that contains the trained models.
            Does not need to be provided in case ``local_result``
            is set to ``True`` during training.
        spark
            Spark session instance. Only provide when ``df`` is a pandas DataFrame.

        Returns
        -------
            DataFrame that contains predictions per time series.
        """
        id_col = self.id_col
        date_col = self.date_col
        input_type = _check_input_type(df)
        _check_fitted(self, trained_models, spark)
        _check_spark(self, input_type, spark)

        @F.pandas_udf(
            "id string, date date, prediction float",
            functionType=F.PandasUDFType.GROUPED_MAP,
        )
        def _predict_udf(df):
            data = pickle.loads(df["data"].iloc[0])
            data = self._convert_categorical(data)
            forecast_horizon_list = df["forecast_horizon"].iloc[0]
            model_list = df["model"].iloc[0]

            result_list = []
            for i in range(self.n_horizon):
                forecast_horizon = forecast_horizon_list[i]
                features = self._filter_features(data, forecast_horizon)
                model_data = self._filter_horizon(data, forecast_horizon)

                model = pickle.loads(model_list[i])
                model_data["prediction"] = model.predict(model_data[features])

                result_list.append(model_data)

            result = pd.concat(result_list).reset_index()

            return result[[id_col, date_col, "prediction"]]

        df = spark.createDataFrame(df) if input_type == "df_pandas" else df
        df = df.withColumn("date", F.to_timestamp("date"))

        trained_models = (
            spark.createDataFrame(self.model_)
            if ((trained_models is None) | (input_type == "df_pandas"))
            else trained_models
        )
        df = self._predict_grid(df, trained_models)

        predictions = df.groupby("group").apply(_predict_udf)
        if input_type == "df_pandas":
            return predictions.toPandas()
        else:
            return predictions
