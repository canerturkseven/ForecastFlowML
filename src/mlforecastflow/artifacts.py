import mlflow
import pandas as pd
from lightgbm import LGBMRegressor
from mlflow.models.signature import infer_signature
import plotly.express as px
from typing import List


class ModelArtifact:
    def __init__(
        self,
        *,
        model_name: str,
        model: LGBMRegressor,
        df_train: pd.DataFrame,
        features: List[str],
        target_col: str,
        forecast_horizon: List[int],
        forecast: pd.DataFrame,
    ):
        self.model_name = model_name
        self.model = model
        self.df_train = df_train
        self.features = features
        self.target_col = target_col
        self.forecast_horizon = forecast_horizon
        self.forecast = forecast

    def log_model(self):
        X, y = self.df_train[self.features], self.df_train[self.target_col]
        self.model.fit(X, y)
        signature = infer_signature(X)
        mlflow.lightgbm.log_model(
            lgb_model=self.model,
            artifact_path=f"models/{self.model_name}",
            signature=signature,
            metadata={"forecast_horizon": self.forecast_horizon},
        )

    def log_feature_importance(self):
        df_importance = pd.DataFrame(
            {
                "feature": self.features,
                "importance": self.model.feature_importances_,
            }
        ).sort_values(by=["importance"])
        graph = px.bar(df_importance, y="feature", x="importance", orientation="h")
        graph.write_html(f"model_{self.model_name}.html")
        mlflow.log_artifact(f"model_{self.model_name}.html", "feature_importance")


class CrossValidationArtifact:
    def __init__(
        self,
        *,
        cv_metrics: pd.DataFrame,
        forecast: pd.DataFrame,
        id_cols: List[str],
        df_train: pd.DataFrame,
        date_col: str,
        target_col: str,
    ):

        self.cv_metrics = cv_metrics
        self.forecast = forecast
        self.id_cols = id_cols
        self.df_train = df_train
        self.date_col = date_col
        self.target_col = target_col

    def log_forecast_graph(self):
        ids = [*self.id_cols, self.date_col]
        forecast = self.forecast.pivot_table(
            index=ids, columns="cv", values="forecast"
        ).reset_index()
        graph = px.line(
            self.df_train[[*ids, self.target_col]]
            .merge(forecast, on=ids, how="left")
            .groupby(self.date_col)
            .sum(min_count=1)
        )
        graph.write_html("forecast_graph.html")
        mlflow.log_artifact("forecast_graph.html")

    def log_forecast(self):
        self.forecast.to_parquet("forecast.parquet")
        mlflow.log_artifact("forecast.parquet")

    def log_cv_metrics(self):
        self.cv_metrics.to_csv("cv_metrics.csv")
        mlflow.log_artifact("cv_metrics.csv")
