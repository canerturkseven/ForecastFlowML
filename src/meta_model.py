import mlflow
from mlflow.models.signature import infer_signature
from mlflow import MlflowClient

import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
)

import re
import pandas as pd
from time_based_split import TimeBasedSplit
from lightgbm import LGBMRegressor
import pyspark.sql.functions as F
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
import plotly.express as px
from collections import namedtuple


class MetaModel(mlflow.pyfunc.PythonModel):
    def __init__(
        self,
        run_id,
        id_cols,
        group_col,
        date_col,
        target_col,
        date_frequency,
        n_cv_splits,
        max_forecast_horizon,
        model_horizon,
        tracking_uri,
        max_hyperparam_evals,
        metric,
        hyperparam_space_fn,
    ):

        self.run_id = run_id
        self.id_cols = id_cols
        self.group_col = group_col
        self.date_col = date_col
        self.target_col = target_col
        self.date_frequency = date_frequency
        self.n_cv_splits = n_cv_splits
        self.max_forecast_horizon = max_forecast_horizon
        self.model_horizon = model_horizon
        self.tracking_uri = tracking_uri
        self.max_hyperparam_evals = max_hyperparam_evals
        self.metric = metric
        self.hyperparam_space_fn = hyperparam_space_fn

    def _filter_features(self, df, forecast_horizon):
        numeric_cols = df.select_dtypes("number").columns
        not_allowed_lags = [f"lag_{i}(_|$)" for i in range(1, max(forecast_horizon))]
        target = [f"^{self.target_col}$"]
        combined_regex = "(" + ")|(".join(not_allowed_lags + target) + ")"
        return [
            col
            for col in numeric_cols
            if not re.search(combined_regex, col, re.IGNORECASE)
        ]

    def _cross_val_forecast(self, model, df, cv, features):
        forecast = []
        for i, fold in enumerate(cv):

            train_idx, test_idx = fold[0], fold[1]
            df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]

            model.fit(df_train[features], df_train[self.target_col])
            df_test["forecast"] = model.predict(df_test[features])
            df_test["cv"] = i
            forecast.append(
                df_test[
                    [*self.id_cols, self.date_col, "cv", self.target_col, "forecast"]
                ]
            )

        return pd.concat(forecast).reset_index(drop=True)

    def _filter_horizon(self, df, forecast_horizon):
        dates = df[self.date_col].sort_values().unique()
        forecast_dates = dates[[fh - 1 for fh in forecast_horizon]]
        return df[df[self.date_col].isin(forecast_dates)]

    def _get_score(self, df):
        supported_metrics = ["mse", "rmse", "mae", "wmape"]
        y_true, y_pred = df[self.target_col], df["forecast"]
        if self.metric == "mse":
            return mean_squared_error(y_true, y_pred)
        elif self.metric == "rmse":
            return mean_squared_error(y_true, y_pred, squared=False)
        elif self.metric == "mae":
            return mean_absolute_error(y_true, y_pred)
        elif self.metric == "wmape":
            return mean_absolute_percentage_error(y_true, y_pred, sample_weight=y_true)
        else:
            ValueError(
                f"Unknown metric: {self.metric}. It must be one of: {', '.join(supported_metrics)}"
            )

    def _evaluate(self, cv_forecast):
        def calculate_metrics(df):
            return pd.Series(
                {
                    "val_start": df[self.date_col].min(),
                    "val_end": df[self.date_col].max(),
                    self.metric: self._get_score(df),
                }
            )

        return cv_forecast.groupby(["cv"]).apply(calculate_metrics)

    @staticmethod
    def _importance_graph(model, features):
        df_importance = pd.DataFrame(
            {"feature": features, "importance": model.feature_importances_}
        ).sort_values(by=["importance"])
        return px.bar(df_importance, y="feature", x="importance", orientation="h")

    def _forecast_graph(self, df, cv_forecast):
        ids = [*self.id_cols, self.date_col]
        cv_forecast = cv_forecast.pivot_table(
            index=ids, columns="cv", values="forecast"
        ).reset_index()
        return px.line(
            df[[*ids, self.target_col]]
            .merge(cv_forecast, on=ids, how="left")
            .groupby(self.date_col)
            .sum(min_count=1)
        )

    # TODO: make it a bit more clearer
    def _best_model_callback(self, study, trial):
        if trial.value <= study.best_value:
            model_artifacts_list = trial.user_attrs.get("model_artifacts_list")
            for i, model_artifacts in enumerate(model_artifacts_list):
                mlflow.lightgbm.log_model(
                    lgb_model=model_artifacts.model,
                    artifact_path=f"models/fh_{i}",
                    signature=model_artifacts.signature,
                    # metadata={"forecast_horizon": forecast_horizon},
                )
                feature_importance_file = f"model_fh_{i}.html"
                model_artifacts.feature_importance.write_html(feature_importance_file)
                mlflow.log_artifact(feature_importance_file, "feature_importance")

        trial_artifacts = trial.user_attrs.get("trial_artifacts")

        forecast_graph_file = "predict_graph.html"
        trial_artifacts.forecast_graph.write_html(forecast_graph_file)
        mlflow.log_artifact(forecast_graph_file)

        forecast_file = "forecast.parquet"
        trial_artifacts.forecast.to_parquet(forecast_file)
        mlflow.log_artifact(forecast_file)

        cv_metric_file = "cv_metric.csv"
        trial_artifacts.cv_metric.to_csv(cv_metric_file)
        mlflow.log_artifact(cv_metric_file)

        mlflow.log_metric(self.metric, trial.value)

    def _optimize_models(self, df):
        def objective_fn(trial):

            model_artifacts = namedtuple(
                "model_artifacts",
                [
                    "model",
                    "forecast_horizon",
                    "feature_importance",
                    "signature",
                    "model_forecast",
                ],
            )
            trial_artifacts = namedtuple(
                "trial_artifacts", ["cv_metric", "forecast", "forecast_graph"]
            )
            n_models = self.max_forecast_horizon // self.model_horizon
            model_artifacts_list = []

            for i in range(n_models):

                model = LGBMRegressor(**self.hyperparam_space_fn(trial))

                forecast_horizon = list(
                    range(i * self.model_horizon + 1, (i + 1) * self.model_horizon + 1)
                )
                features = self._filter_features(
                    df=df, forecast_horizon=forecast_horizon
                )
                cv = TimeBasedSplit(
                    date_col=self.date_col,
                    date_frequency=self.date_frequency,
                    n_splits=self.n_cv_splits,
                    forecast_horizon=forecast_horizon,
                    step_length=self.max_forecast_horizon,
                    end_offset=self.max_forecast_horizon - max(forecast_horizon),
                )
                model_forecast = self._cross_val_forecast(
                    model=model, df=df, cv=cv.split(df), features=features
                )

                model.fit(df[features], df[self.target_col])
                feature_importance = self._importance_graph(model, features)
                signature = infer_signature(df[features])

                model_artifacts_list.append(
                    model_artifacts(
                        model,
                        forecast_horizon,
                        feature_importance,
                        signature,
                        model_forecast,
                    )
                )

            forecast = pd.concat(
                [
                    model_artifacts.model_forecast
                    for model_artifacts in model_artifacts_list
                ]
            )
            forecast_graph = self._forecast_graph(df, forecast)
            cv_metric = self._evaluate(forecast)

            trial.set_user_attr(
                "trial_artifacts", trial_artifacts(cv_metric, forecast, forecast_graph)
            )
            trial.set_user_attr("model_artifacts_list", model_artifacts_list)

            return cv_metric[self.metric].mean()

        study = optuna.create_study(direction="minimize")
        study.optimize(
            objective_fn,
            n_trials=self.max_hyperparam_evals,
            callbacks=[self._best_model_callback],
        )
        self._plot_optimization_study(study)

    @staticmethod
    def _plot_optimization_study(study):
        for func, output_file in [
            (plot_optimization_history, "optimize_history.html"),
            (plot_parallel_coordinate, "parallel_coordinate.html"),
            (plot_param_importances, "param_importances.html"),
        ]:
            func(study).write_html(output_file)
            mlflow.log_artifact(output_file)

    def train(self, df):
        run_id = self.run_id
        group_col = self.group_col
        tracking_uri = self.tracking_uri

        @F.pandas_udf("status string", functionType=F.PandasUDFType.GROUPED_MAP)
        def train_udf(df):
            mlflow.set_tracking_uri(tracking_uri)
            group_name = df[group_col].iloc[0]

            with mlflow.start_run(run_id=run_id):
                with mlflow.start_run(run_name=group_name, nested=True):
                    self._optimize_models(df)

            return pd.DataFrame([{"status": "ok"}])

        df.write.parquet("train.parquet")
        mlflow.log_artifact("train.parquet")
        (
            df.withColumn("run_id", F.lit(self.run_id))
            .groupby(self.group_col)
            .apply(train_udf)
            .collect()
        )

    def predict(self, context, model_input):
        tracking_uri = self.tracking_uri
        group_col = self.group_col
        run_id = self.run_id
        id_cols = self.id_cols
        date_col = self.date_col
        schema = ", ".join(
            [
                *[f"{col} string" for col in self.id_cols],
                f"{date_col} date",
                "prediction double",
            ]
        )

        @F.pandas_udf(
            schema,
            functionType=F.PandasUDFType.GROUPED_MAP,
        )
        def predict_udf(df):
            mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient()

            group_name = df[group_col].iloc[0]
            model_uris = [
                f"runs:/{run_id}/{model.path}"
                for model in client.list_artifacts(
                    run_id=run_id, path=f"models/{group_name}"
                )
            ]

            df_forecast = []
            for model_uri in model_uris:
                forecast_horizon = list(
                    map(int, model_uri.split("/")[-1].split("_")[1:])
                )
                model_info = mlflow.models.get_model_info(model_uri)
                features = model_info.signature.inputs.input_names()
                df_model = self._filter_horizon(df, forecast_horizon)

                model = mlflow.lightgbm.load_model(model_uri)
                df_model["prediction"] = model.predict(df_model[features])
                df_forecast.append(df_model[[*id_cols, date_col, "prediction"]])

            return pd.concat(df_forecast)

        return model_input.groupBy(group_col).apply(predict_udf)
