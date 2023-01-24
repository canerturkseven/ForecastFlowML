import mlflow
from mlflow import MlflowClient
import pandas as pd
import pyspark.sql.functions as F
from mlforecastflow.optimizer import Optimizer


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

    def _filter_horizon(self, df, forecast_horizon):
        dates = df[self.date_col].sort_values().unique()
        forecast_dates = dates[[fh - 1 for fh in forecast_horizon]]
        return df[df[self.date_col].isin(forecast_dates)]

    def train(self, df):
        id_cols = self.id_cols
        date_col = self.date_col
        target_col = self.target_col
        max_forecast_horizon = self.max_forecast_horizon
        model_horizon = self.model_horizon
        date_frequency = self.date_frequency
        metric = self.metric
        max_hyperparam_evals = self.max_hyperparam_evals
        n_cv_splits = self.n_cv_splits
        hyperparam_space_fn = self.hyperparam_space_fn
        run_id = self.run_id
        group_col = self.group_col
        tracking_uri = self.tracking_uri

        @F.pandas_udf("status string", functionType=F.PandasUDFType.GROUPED_MAP)
        def train_udf(df):
            mlflow.set_tracking_uri(tracking_uri)
            group_name = df[group_col].iloc[0]
            with mlflow.start_run(run_id=run_id):
                with mlflow.start_run(run_name=group_name, nested=True):

                    df.to_parquet("train.parquet")
                    mlflow.log_artifact("train.parquet")

                    optimizer = Optimizer(
                        id_cols=id_cols,
                        date_col=date_col,
                        target_col=target_col,
                        max_forecast_horizon=max_forecast_horizon,
                        model_horizon=model_horizon,
                        hyperparam_space_fn=hyperparam_space_fn,
                        date_frequency=date_frequency,
                        n_cv_splits=n_cv_splits,
                        metric=metric,
                        max_hyperparam_evals=max_hyperparam_evals,
                    )
                    optimizer.run(df)

            return pd.DataFrame([{"status": "ok"}])

        (
            df.withColumn("run_id", F.lit(self.run_id))
            .groupby(self.group_col)
            .apply(train_udf)
            .collect()
        )

    def predict(self, context, model_input):
        tracking_uri = self.tracking_uri
        group_col = self.group_col
        parent_run_id = self.run_id
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
            filter_string = (
                f"tags.mlflow.parentRunId = '{parent_run_id}'"
                f"and tags.mlflow.runName = '{group_name}'"
            )
            run_id = mlflow.search_runs(filter_string=filter_string)["run_id"].iloc[0]

            model_uris = [
                f"runs:/{run_id}/{model.path}"
                for model in client.list_artifacts(run_id=run_id, path=f"models")
            ]

            df_forecast = []
            for model_uri in model_uris:

                model = mlflow.lightgbm.load_model(model_uri)
                model_info = mlflow.models.get_model_info(model_uri)
                forecast_horizon = model_info.metadata["forecast_horizon"]
                features = model_info.signature.inputs.input_names()

                df_model = self._filter_horizon(df, forecast_horizon)
                df_model["prediction"] = model.predict(df_model[features])

                df_forecast.append(df_model[[*id_cols, date_col, "prediction"]])

            return pd.concat(df_forecast)

        return model_input.groupBy(group_col).apply(predict_udf)
