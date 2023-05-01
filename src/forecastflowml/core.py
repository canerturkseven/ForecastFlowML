import os
import pickle
import datetime
import sklearn
import pyspark
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from forecastflowml.model_selection import (
    _cross_val_predict,
    _score_func,
    _TimeBasedSplit,
)
from forecastflowml.utils import _check_input_type, _check_spark, _check_fitted
from forecastflowml.direct_forecaster import _DirectForecaster
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
            if pyspark.__version__ <= "2.4":
                os.environ["ARROW_PRE_0_15_IPC_FORMAT"] = "1"

            group = df["group"].iloc[0]

            importance_list = []
            for i in range(len(df["model"].iloc[0])):
                model = pickle.loads(bytes(df["model"].iloc[0][i], "latin1"))
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
            schema = (
                "group:string, forecast_horizon:array<int>, "
                "feature:string, importance:float"
            )
            if pyspark.__version__ < "3":
                pandas_udf = F.pandas_udf(
                    _feature_importance_udf,
                    schema=schema,
                    functionType=F.PandasUDFType.GROUPED_MAP,
                )
                return df_model.groupby("group").apply(pandas_udf).toPandas()
            else:
                return (
                    df_model.groupby("group")
                    .applyInPandas(_feature_importance_udf, schema=schema)
                    .toPandas()
                )
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
        id_col = self.id_col
        date_col = self.date_col
        categorical_cols = self.categorical_cols
        model_horizon = self.model_horizon
        group_col = self.group_col
        target_col = self.target_col
        max_forecast_horizon = self.max_forecast_horizon
        use_lag_range = self.use_lag_range
        model = self.model
        input_type = _check_input_type(df)
        _check_spark(self, input_type, spark)

        def _train_udf(df):
            # Ensure pyarrow compatibility with old versions
            if pyspark.__version__ <= "2.4":
                os.environ["ARROW_PRE_0_15_IPC_FORMAT"] = "1"

            start = datetime.datetime.now()

            forecaster = _DirectForecaster(
                id_col=id_col,
                group_col=group_col,
                date_col=date_col,
                target_col=target_col,
                categorical_cols=categorical_cols,
                model=model,
                model_horizon=model_horizon,
                max_forecast_horizon=max_forecast_horizon,
                use_lag_range=use_lag_range,
            )
            forecaster.fit(df)

            end = datetime.datetime.now()
            elapsed = end - start
            seconds = round(elapsed.total_seconds(), 1)

            return pd.DataFrame(
                [
                    {
                        "group": df[group_col].iloc[0],
                        "forecast_horizon": [list(x) for x in forecaster.model_.keys()],
                        "model": [
                            str(pickle.dumps(x), "latin1")
                            for x in forecaster.model_.values()
                        ],
                        "start_time": start.strftime("%d-%b-%Y (%H:%M:%S)"),
                        "end_time": end.strftime("%d-%b-%Y (%H:%M:%S)"),
                        "elapsed_seconds": seconds,
                    },
                ]
            )

        df = spark.createDataFrame(df) if input_type == "df_pandas" else df
        df = df.withColumn("date", F.to_timestamp("date"))

        schema = (
            "group:string, forecast_horizon:array<array<int>>, model:array<string>,"
            "start_time:string, end_time:string, elapsed_seconds:float"
        )
        if pyspark.__version__ < "3":
            pandas_udf = F.pandas_udf(
                _train_udf, schema=schema, functionType=F.PandasUDFType.GROUPED_MAP
            )
            model_ = df.groupby(group_col).apply(pandas_udf)
        else:
            model_ = df.groupby(group_col).applyInPandas(_train_udf, schema=schema)

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
        categorical_cols = self.categorical_cols
        model_horizon = self.model_horizon
        date_col = self.date_col
        date_frequency = self.date_frequency
        max_forecast_horizon = self.max_forecast_horizon
        group_col = self.group_col
        use_lag_range = self.use_lag_range
        model = self.model
        cv_step_length = (
            max_forecast_horizon if cv_step_length is None else cv_step_length
        )
        input_type = _check_input_type(df)
        _check_spark(self, input_type, spark)

        def _cross_validate_udf(df):
            # Ensure pyarrow compatibility with old versions
            if pyspark.__version__ <= "2.4":
                os.environ["ARROW_PRE_0_15_IPC_FORMAT"] = "1"

            forecaster = _DirectForecaster(
                id_col=id_col,
                group_col=group_col,
                date_col=date_col,
                target_col=target_col,
                categorical_cols=categorical_cols,
                model=model,
                model_horizon=model_horizon,
                max_forecast_horizon=max_forecast_horizon,
                use_lag_range=use_lag_range,
            )

            cv = _TimeBasedSplit(
                date_col=date_col,
                date_frequency=date_frequency,
                n_splits=int(n_cv_splits),
                forecast_horizon=list(range(1, max_forecast_horizon + 1)),
                step_length=int(cv_step_length),
                max_train_size=max_train_size,
            ).split(df)

            cv_predictions = _cross_val_predict(
                forecaster=forecaster,
                df=df,
                cv=cv,
                refit=refit,
            )

            return cv_predictions

        df = spark.createDataFrame(df) if input_type == "df_pandas" else df
        df = df.withColumn("date", F.to_timestamp("date"))

        schema = (
            "group string, id string, date date, cv string,"
            "target float, prediction float"
        )
        if pyspark.__version__ < "3":
            pandas_udf = F.pandas_udf(
                _cross_validate_udf,
                schema=schema,
                functionType=F.PandasUDFType.GROUPED_MAP,
            )
            cv_result = df.groupby(group_col).apply(pandas_udf)
        else:
            cv_result = df.groupby(group_col).applyInPandas(
                _cross_validate_udf, schema=schema
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
        max_train_size: Optional[int] = None,
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
        max_train_size
            Number of max periods to use as training set.
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
        categorical_cols = self.categorical_cols
        model_horizon = self.model_horizon
        use_lag_range = self.use_lag_range
        max_forecast_horizon = self.max_forecast_horizon
        cv_step_length = (
            max_forecast_horizon if cv_step_length is None else cv_step_length
        )
        input_type = _check_input_type(df)
        _check_spark(self, input_type, spark)

        def _grid_search_udf(df):
            # Ensure pyarrow compatibility with old versions
            if pyspark.__version__ <= "2.4":
                os.environ["ARROW_PRE_0_15_IPC_FORMAT"] = "1"

            group = df[group_col].iloc[0]
            hyperparams = {param: df[param].iloc[0] for param in param_grid.keys()}
            try_model = model.set_params(**hyperparams)

            forecaster = _DirectForecaster(
                id_col=id_col,
                group_col=group_col,
                date_col=date_col,
                target_col=target_col,
                categorical_cols=categorical_cols,
                model=try_model,
                model_horizon=model_horizon,
                max_forecast_horizon=max_forecast_horizon,
                use_lag_range=use_lag_range,
            )

            cv = _TimeBasedSplit(
                date_col=date_col,
                date_frequency=date_frequency,
                n_splits=int(n_cv_splits),
                forecast_horizon=list(range(1, max_forecast_horizon + 1)),
                step_length=int(cv_step_length),
                max_train_size=max_train_size,
            ).split(df)

            cv_predictions = _cross_val_predict(
                forecaster=forecaster,
                df=df,
                cv=cv,
                refit=refit,
            )

            score = (
                cv_predictions.groupby("cv")
                .apply(
                    lambda x: _score_func(x["target"], x["prediction"], scoring_metric)
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

        schema = "group string, score float, " + ", ".join(
            [f"{key} {type(value[0]).__name__}" for key, value in param_grid.items()]
        )
        if pyspark.__version__ < "3":
            pandas_udf = F.pandas_udf(
                _grid_search_udf,
                schema=schema,
                functionType=F.PandasUDFType.GROUPED_MAP,
            )
            cv_result = df.groupby([group_col, *param_grid.keys()]).apply(pandas_udf)
        else:
            cv_result = df.groupby([group_col, *param_grid.keys()]).applyInPandas(
                _grid_search_udf, schema=schema
            )

        return (
            cv_result.toPandas()
            .sort_values(by=["group", "score"], ascending=False)
            .reset_index(drop=True)
        )

    def _serialize(self, df):
        group_col = self.group_col

        def _serialize_udf(df):
            return pd.DataFrame(
                [
                    {
                        "group": df[group_col].iloc[0],
                        "data": str(pickle.dumps(df), "latin1"),
                    }
                ]
            )

        schema = "group:string, data:string"
        if pyspark.__version__ < "3":
            pandas_udf = F.pandas_udf(
                _serialize_udf,
                schema=schema,
                functionType=F.PandasUDFType.GROUPED_MAP,
            )
            return df.groupby(group_col).apply(pandas_udf)
        else:
            return df.groupby(group_col).applyInPandas(_serialize_udf, schema=schema)

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
        group_col = self.group_col
        date_col = self.date_col
        target_col = self.target_col
        categorical_cols = self.categorical_cols
        model = self.model
        model_horizon = self.model_horizon
        max_forecast_horizon = self.max_forecast_horizon
        use_lag_range = self.use_lag_range
        input_type = _check_input_type(df)
        _check_fitted(self, trained_models, spark)
        _check_spark(self, input_type, spark)

        def _predict_udf(df):
            if pyspark.__version__ <= "2.4":
                os.environ["ARROW_PRE_0_15_IPC_FORMAT"] = "1"

            data = pickle.loads(bytes(df["data"].iloc[0], "latin1"))
            forecast_horizon_list = list(map(tuple, df["forecast_horizon"].iloc[0]))
            model_list = [pickle.loads(bytes(m, "latin1")) for m in df["model"].iloc[0]]
            model_ = {fh: model for fh, model in zip(forecast_horizon_list, model_list)}

            forecaster = _DirectForecaster(
                id_col=id_col,
                group_col=group_col,
                date_col=date_col,
                target_col=target_col,
                categorical_cols=categorical_cols,
                model=model,
                model_horizon=model_horizon,
                max_forecast_horizon=max_forecast_horizon,
                use_lag_range=use_lag_range,
            )
            forecaster.model_ = model_
            prediction = forecaster.predict(data)

            return prediction

        df = spark.createDataFrame(df) if input_type == "df_pandas" else df
        df = df.withColumn("date", F.to_timestamp("date"))

        trained_models = (
            spark.createDataFrame(self.model_)
            if ((trained_models is None) | (input_type == "df_pandas"))
            else trained_models
        )
        df = self._predict_grid(df, trained_models)

        schema = "group:string, id:string, date:date, prediction:float"
        if pyspark.__version__ < "3":
            pandas_udf = F.pandas_udf(
                _predict_udf, schema=schema, functionType=F.PandasUDFType.GROUPED_MAP
            )
            predictions = df.groupby("group").apply(pandas_udf)
        else:
            predictions = df.groupby("group").applyInPandas(_predict_udf, schema=schema)

        if input_type == "df_pandas":
            return predictions.toPandas()
        else:
            return predictions
