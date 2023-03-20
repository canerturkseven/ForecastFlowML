import tempfile
import os
import shutil
import mlflow
import sklearn
from mlflow.models.signature import infer_signature
import pandas as pd
import plotly.express as px
from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
)

##TODO: Fully make sure that written artifacts do not collide with each other.
##TODO: Make sure they have unique path, if not throw error.
##TODO: Otherwise, it is very hard to find error inside of Pandas UDF.


def log_train_data(group_name, df, horizon_id):
    tempdir = tempfile.mkdtemp()
    print(
        f"logging train data, dir: {tempdir}, group: {group_name}, horizon: {horizon_id}"
    )
    try:
        filepath = os.path.join(tempdir, f"horizon_{horizon_id}.parquet")
        df.to_parquet(filepath)
        mlflow.log_artifact(filepath, "train_data")
    finally:
        shutil.rmtree(tempdir)


def log_model(df_train, model, horizon_id):
    signature = infer_signature(df_train[model.feature_name_])
    mlflow.lightgbm.log_model(
        lgb_model=model,
        artifact_path=f"models/horizon_{horizon_id}",
        signature=signature,
    )


def log_feature_importance(group_name, model, horizon_id):
    df_importance = pd.DataFrame(
        {
            "feature": model.feature_name_,
            "importance": model.feature_importances_,
        }
    ).sort_values(by=["importance"])
    graph = px.bar(df_importance, y="feature", x="importance", orientation="h")
    tempdir = tempfile.mkdtemp()
    print(
        f"logging feature importance, dir: {tempdir}, group: {group_name}, horizon: {horizon_id}"
    )
    try:
        html_path = os.path.join(tempdir, f"horizon_{horizon_id}.html")
        json_path = os.path.join(tempdir, f"horizon_{horizon_id}.json")
        graph.write_html(html_path)
        graph.write_json(json_path)
        mlflow.log_artifact(html_path, "feature_importance")
        mlflow.log_artifact(json_path, "feature_importance")
    finally:
        shutil.rmtree(tempdir)


def log_optimization_visualisation(group_name, study, horizon_id):
    for func, folder in [
        (plot_optimization_history, "optimize_history"),
        (plot_parallel_coordinate, "parallel_coordinate"),
        (plot_param_importances, "param_importances"),
    ]:
        tempdir = tempfile.mkdtemp()
        print(
            f"logging optimise graph, dir: {tempdir}, group: {group_name}, horizon: {horizon_id}"
        )
        try:
            graph = func(study)
            html_path = os.path.join(tempdir, f"horizon_{horizon_id}.html")
            json_path = os.path.join(tempdir, f"horizon_{horizon_id}.json")
            graph.write_html(html_path)
            graph.write_json(json_path)
            mlflow.log_artifact(html_path, f"optimisation_visualization/{folder}")
            mlflow.log_artifact(json_path, f"optimisation_visualization/{folder}")
        except:
            pass
        finally:
            shutil.rmtree(tempdir)


def log_cv_forecast(group_name, cv_forecast, horizon_id):
    tempdir = tempfile.mkdtemp()
    print(
        f"logging cv forecast, dir: {tempdir}, group: {group_name}, horizon: {horizon_id}"
    )
    try:
        filepath = os.path.join(tempdir, f"horizon_{horizon_id}.parquet")
        cv_forecast.to_parquet(filepath)
        mlflow.log_artifact(filepath, "cv_forecast")
    finally:
        shutil.rmtree(tempdir)


def log_metric(group_run_ids, target_col, scoring_metric):
    def score_func(y_true, y_pred):
        sklearn_scorer = sklearn.metrics.get_scorer(scoring_metric)
        return sklearn_scorer._sign * sklearn_scorer._score_func(
            y_true=y_true, y_pred=y_pred, **sklearn_scorer._kwargs
        )

    for group_run_id in group_run_ids.values():
        with mlflow.start_run(run_id=group_run_id):
            artifact_uri = mlflow.artifacts.download_artifacts(run_id=group_run_id)
            cv_forecast = pd.read_parquet(os.path.join(artifact_uri, "cv_forecast"))
            score = (
                cv_forecast.groupby("cv")
                .apply(
                    lambda x: score_func(
                        x[target_col].to_numpy(), x["forecast"].to_numpy()
                    )
                )
                .mean()
            )
            mlflow.log_metric(scoring_metric, score)


def log_cv_forecast_graph(group_run_ids, id_cols, target_col, date_col):
    for group_name, group_run_id in group_run_ids.items():
        with mlflow.start_run(run_id=group_run_id):
            artifact_uri = mlflow.artifacts.download_artifacts(run_id=group_run_id)
            train = pd.read_parquet(os.path.join(artifact_uri, "train_data"))
            cv_forecast = (
                pd.read_parquet(os.path.join(artifact_uri, "cv_forecast"))
                .pivot_table(
                    index=[*id_cols, date_col],
                    columns="cv",
                    values="forecast",
                )
                .reset_index()
            )
            graph = px.line(
                train[[*id_cols, date_col, target_col]]
                .merge(cv_forecast, on=[*id_cols, date_col], how="left")
                .groupby(date_col)
                .sum(min_count=1)
            )
            tempdir = tempfile.mkdtemp()
            try:
                html_path = os.path.join(tempdir, f"cv_forecast_graph.html")
                json_path = os.path.join(tempdir, f"cv_forecast_graph.json")
                graph.write_html(html_path)
                graph.write_json(json_path)
                mlflow.log_artifact(html_path, f"cv_forecast_graph")
                mlflow.log_artifact(json_path, f"cv_forecast_graph")
            finally:
                shutil.rmtree(tempdir)
