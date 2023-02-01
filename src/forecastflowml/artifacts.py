import mlflow
import pandas as pd
import plotly.express as px
from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
)


class Artifact:
    def __init__(
        self, *, horizon_id, model, signature, forecast_horizon, study, forecast
    ):
        self.horizon_id = horizon_id
        self.model = model
        self.signature = signature
        self.forecast_horizon = forecast_horizon
        self.study = study
        self.forecast = forecast

    def log_model(self):
        mlflow.lightgbm.log_model(
            lgb_model=self.model,
            artifact_path=f"models/{self.horizon_id}",
            signature=self.signature,
            metadata={"forecast_horizon": self.forecast_horizon},
        )

    def log_feature_importance(self):
        df_importance = pd.DataFrame(
            {
                "feature": self.model.feature_name_,
                "importance": self.model.feature_importances_,
            }
        ).sort_values(by=["importance"])
        graph = px.bar(df_importance, y="feature", x="importance", orientation="h")
        graph.write_html(f"{self.horizon_id}.html")
        mlflow.log_artifact(f"{self.horizon_id}.html", "feature_importance")

    def log_optimization_visualisation(self):
        for func, folder in [
            (plot_optimization_history, "optimize_history"),
            (plot_parallel_coordinate, "parallel_coordinate"),
            (plot_param_importances, "param_importances"),
        ]:
            try:
                func(self.study).write_html(f"{self.horizon_id}.html")
                mlflow.log_artifact(
                    f"{self.horizon_id}.html", f"optimisation_visualization/{folder}"
                )
            except:
                pass

    def log_forecast(self):
        self.forecast.to_parquet(f"{self.horizon_id}.parquet")
        mlflow.log_artifact(f"{self.horizon_id}.parquet", f"cv_forecast")
