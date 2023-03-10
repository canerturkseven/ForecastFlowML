import tempfile
import os
import shutil
import mlflow
from mlflow.models.signature import infer_signature
import pandas as pd
import plotly.express as px
from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
)


class Artifact:
    def __init__(
        self,
        *,
        df_train,
        horizon_id,
        model,
        study,
        cv_forecast,
    ):
        self.df_train = df_train
        self.horizon_id = horizon_id
        self.model = model
        self.study = study
        self.cv_forecast = cv_forecast

    def log_train_data(self):
        tempdir = tempfile.mkdtemp()
        try:
            filepath = os.path.join(tempdir, f"horizon_{self.horizon_id}.parquet")
            self.df_train.to_parquet(filepath)
            mlflow.log_artifact(filepath, "train_data")
        finally:
            shutil.rmtree(tempdir)

    def log_model(self):
        signature = infer_signature(self.df_train[self.model.feature_name_])
        mlflow.lightgbm.log_model(
            lgb_model=self.model,
            artifact_path=f"models/horizon_{self.horizon_id}",
            signature=signature,
        )

    def log_feature_importance(self):
        df_importance = pd.DataFrame(
            {
                "feature": self.model.feature_name_,
                "importance": self.model.feature_importances_,
            }
        ).sort_values(by=["importance"])
        graph = px.bar(df_importance, y="feature", x="importance", orientation="h")
        tempdir = tempfile.mkdtemp()
        try:
            html_path = os.path.join(tempdir, f"horizon_{self.horizon_id}.html")
            json_path = os.path.join(tempdir, f"horizon_{self.horizon_id}.json")
            graph.write_html(html_path)
            graph.write_json(json_path)
            mlflow.log_artifact(html_path, "feature_importance")
            mlflow.log_artifact(json_path, "feature_importance")
        finally:
            shutil.rmtree(tempdir)

    def log_optimization_visualisation(self):
        for func, folder in [
            (plot_optimization_history, "optimize_history"),
            (plot_parallel_coordinate, "parallel_coordinate"),
            (plot_param_importances, "param_importances"),
        ]:
            tempdir = tempfile.mkdtemp()
            try:
                graph = func(self.study)
                html_path = os.path.join(tempdir, f"horizon_{self.horizon_id}.html")
                json_path = os.path.join(tempdir, f"horizon_{self.horizon_id}.json")
                graph.write_html(html_path)
                graph.write_json(json_path)
                mlflow.log_artifact(html_path, f"optimisation_visualization/{folder}")
                mlflow.log_artifact(json_path, f"optimisation_visualization/{folder}")
            except:
                pass
            finally:
                shutil.rmtree(tempdir)

    def log_cv_forecast(self):
        tempdir = tempfile.mkdtemp()
        try:
            filepath = os.path.join(tempdir, f"horizon_{self.horizon_id}.parquet")
            self.cv_forecast.to_parquet(filepath)
            mlflow.log_artifact(filepath, "cv_forecast")
        finally:
            shutil.rmtree(tempdir)
