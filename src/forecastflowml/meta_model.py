import pickle
import re
import datetime
import pandas as pd
import pyspark.sql.functions as F
from forecastflowml.model_selection import cross_val_forecast, score_func
from forecastflowml.time_based_split import TimeBasedSplit


class ForecastFlowML:
    def __init__(
        self,
        id_col,
        group_col,
        date_col,
        target_col,
        date_frequency,
        max_forecast_horizon,
        model_horizon,
        model,
        categorical_cols=None,
        hyperparams=None,
        use_lag_range=0,
    ):
        self.id_col = id_col
        self.group_col = group_col
        self.date_col = date_col
        self.target_col = target_col
        self.categorical_cols = categorical_cols
        self.date_frequency = date_frequency
        self.model = model
        self.hyperparams = hyperparams if hyperparams is not None else {}
        self.max_forecast_horizon = max_forecast_horizon
        self.model_horizon = model_horizon
        self.use_lag_range = use_lag_range
        self.n_horizon = max_forecast_horizon // model_horizon

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
            f"group string, data binary",
            functionType=F.PandasUDFType.GROUPED_MAP,
        )
        def _serialize_udf(df):
            return pd.DataFrame(
                [{"group": df[group_col].iloc[0], "data": pickle.dumps(df)}]
            )

        return df.groupby(group_col).apply(_serialize_udf)

    def train(self, df):
        group_col = self.group_col
        target_col = self.target_col
        model = self.model
        hyperparams = self.hyperparams

        @F.pandas_udf(
            (
                "group:string, forecast_horizon:array<array<int>>, model:array<binary>,"
                "start_time:string, end_time:string, elapsed_seconds:float"
            ),
            functionType=F.PandasUDFType.GROUPED_MAP,
        )
        def _train_udf(df):
            start = datetime.datetime.now()

            df = self._convert_categorical(df)
            group = df[group_col].iloc[0]
            group_model = (
                model if hyperparams == {} else model.set_params(**hyperparams[group])
            )

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

        return (
            df.withColumn("date", F.to_timestamp("date"))
            .groupby(group_col)
            .apply(_train_udf)
        )

    def cross_validate(self, df, n_cv_splits, cv_step_length):
        id_col = self.id_col
        target_col = self.target_col
        date_col = self.date_col
        date_frequency = self.date_frequency
        max_forecast_horizon = self.max_forecast_horizon
        group_col = self.group_col
        model = self.model
        hyperparams = self.hyperparams

        @F.pandas_udf(
            (
                "group string, id string, date date, cv string,"
                "target float, forecast float"
            ),
            functionType=F.PandasUDFType.GROUPED_MAP,
        )
        def _cross_validate_udf(df):

            df = self._convert_categorical(df)
            group = df[group_col].iloc[0]
            group_model = (
                model if hyperparams == {} else model.set_params(**hyperparams[group])
            )

            cv_forecast_list = []
            for i in range(self.n_horizon):

                forecast_horizon = self._forecast_horizon(i)
                features = self._filter_features(df, forecast_horizon)

                cv = TimeBasedSplit(
                    date_col=date_col,
                    date_frequency=date_frequency,
                    n_splits=int(n_cv_splits),
                    forecast_horizon=list(forecast_horizon),
                    step_length=int(cv_step_length),
                    end_offset=int(max_forecast_horizon - int(max(forecast_horizon))),
                ).split(df)

                cv_forecast = cross_val_forecast(
                    model=group_model,
                    df=df,
                    id_col=id_col,
                    feature_cols=features,
                    date_col=date_col,
                    target_col=target_col,
                    cv=cv,
                )
                cv_forecast_list.append(cv_forecast)

            cv_forecast = pd.concat(cv_forecast_list).reset_index(drop=True)
            cv_forecast.insert(0, "group", group)

            return cv_forecast

        return (
            df.withColumn("date", F.to_timestamp("date"))
            .groupby(group_col)
            .apply(_cross_validate_udf)
        )

    def grid_search(
        self,
        df,
        param_grid,
        n_cv_splits,
        cv_step_length,
        scoring_metric="neg_mean_squared_error",
    ):
        group_col = self.group_col
        id_col = self.id_col
        model = self.model
        target_col = self.target_col
        date_col = self.date_col
        date_frequency = self.date_frequency
        max_forecast_horizon = self.max_forecast_horizon

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

                cv = TimeBasedSplit(
                    date_col=date_col,
                    date_frequency=date_frequency,
                    n_splits=int(n_cv_splits),
                    forecast_horizon=list(forecast_horizon),
                    step_length=int(cv_step_length),
                    end_offset=int(max_forecast_horizon - int(max(forecast_horizon))),
                ).split(df)

                cv_forecast = cross_val_forecast(
                    model=group_model,
                    df=df,
                    id_col=id_col,
                    feature_cols=features,
                    date_col=date_col,
                    target_col=target_col,
                    cv=cv,
                )
                cv_forecast_list.append(cv_forecast)

            cv_forecast = pd.concat(cv_forecast_list)
            score = (
                cv_forecast.groupby("cv")
                .apply(lambda x: score_func(x["target"], x["forecast"], scoring_metric))
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
            .reset_index()
        )

    def _predict_grid(self, df, trained_models):

        df = self._serialize(df)
        df = df.join(
            trained_models.select("group", "forecast_horizon", "model"),
            on="group",
            how="left",
        )
        return df

    def predict(self, df, trained_models):
        id_col = self.id_col
        date_col = self.date_col

        @F.pandas_udf(
            f"id string, date date, prediction float",
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

        df = df.withColumn("date", F.to_timestamp("date"))
        df = self._predict_grid(df, trained_models)

        return df.groupby("group").apply(_predict_udf)
