import tempfile
import os
import shutil
import sklearn
import mlflow
import pandas as pd
import plotly.express as px


class Evaluator:
    def __init__(self, group_run_ids, id_cols, date_col, target_col, scoring):
        self.group_run_ids = group_run_ids
        self.scoring = scoring
        self.id_cols = id_cols
        self.date_col = date_col
        self.target_col = target_col

    def _score(self, y_true, y_pred):
        sklearn_scorer = sklearn.metrics.get_scorer(self.scoring)
        return sklearn_scorer._sign * sklearn_scorer._score_func(
            y_true=y_true, y_pred=y_pred, **sklearn_scorer._kwargs
        )

    def log_metric(self):
        for group_run_id in self.group_run_ids.values():
            with mlflow.start_run(run_id=group_run_id):
                artifact_uri = mlflow.artifacts.download_artifacts(run_id=group_run_id)
                cv_forecast = pd.read_parquet(os.path.join(artifact_uri, "cv_forecast"))
                score = (
                    cv_forecast.groupby("cv")
                    .apply(lambda x: self._score(x[self.target_col], x["forecast"]))
                    .mean()
                )
                mlflow.log_metric(self.scoring, score)

    def log_forecast_graph(self):
        for group_run_id in self.group_run_ids.values():
            with mlflow.start_run(run_id=group_run_id):
                artifact_uri = mlflow.artifacts.download_artifacts(run_id=group_run_id)
                train = pd.read_parquet(os.path.join(artifact_uri, "train_data"))
                cv_forecast = (
                    pd.read_parquet(os.path.join(artifact_uri, "cv_forecast"))
                    .pivot_table(
                        index=[*self.id_cols, self.date_col],
                        columns="cv",
                        values="forecast",
                    )
                    .reset_index()
                )
                graph = px.line(
                    train[[*self.id_cols, self.date_col, self.target_col]]
                    .merge(cv_forecast, on=[*self.id_cols, self.date_col], how="left")
                    .groupby(self.date_col)
                    .sum(min_count=1)
                )
                tempdir = tempfile.mkdtemp()
                try:
                    filepath = os.path.join(tempdir, "cv_forecast_graph.html")
                    graph.write_html(filepath)
                    mlflow.log_artifact(filepath)
                finally:
                    shutil.rmtree(tempdir)
