import mlflow
import re
import os
import functools
import plotly
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession, DataFrame
from forecastflowml.optimizer import Optimizer
from forecastflowml.evaluator import Evaluator


class MetaModel(mlflow.pyfunc.PythonModel):
    def __init__(
        self,
        id_cols,
        group_col,
        date_col,
        target_col,
        date_frequency,
        max_forecast_horizon,
        model_horizon,
        hyperparam_space_fn,
        max_hyperparam_evals=1,
        n_cv_splits=1,
        scoring="neg_mean_squared_error",
        tracking_uri="./mlruns",
        lag_feature_range=0,
        n_jobs=1,
        cv_step_length=None,
    ):
        self.id_cols = id_cols
        self.group_col = group_col
        self.date_col = date_col
        self.target_col = target_col
        self.date_frequency = date_frequency
        self.n_cv_splits = n_cv_splits
        self.max_forecast_horizon = max_forecast_horizon
        self.cv_step_length = (
            cv_step_length if cv_step_length is not None else max_forecast_horizon
        )
        self.model_horizon = model_horizon
        self.tracking_uri = tracking_uri
        self.lag_feature_range = lag_feature_range
        self.max_hyperparam_evals = max_hyperparam_evals
        self.scoring = scoring
        self.hyperparam_space_fn = hyperparam_space_fn
        self.n_jobs = n_jobs
        self.n_horizon = max_forecast_horizon // model_horizon
        mlflow.set_tracking_uri(tracking_uri)

    @property
    def n_horizon(self):
        return self._n_horizon

    @n_horizon.setter
    def n_horizon(self, value):
        self._n_horizon = value

    @property
    def estimators(self):
        est_dict = {}
        for group_name, run_id in self.group_run_ids.items():
            group_dict = {}
            for i in range(self.n_horizon):
                model_path = f"runs:/{run_id}/models/horizon_{i}"
                model = mlflow.lightgbm.load_model(model_path)
                group_dict[f"horizon_{i}"] = model
            est_dict[group_name] = group_dict
        return est_dict

    @property
    def features(self):
        feature_dict = {}
        for group_name, run_id in self.group_run_ids.items():
            group_dict = {}
            for i in range(self.n_horizon):
                model_path = f"runs:/{run_id}/models/horizon_{i}"
                model_info = mlflow.models.get_model_info(model_path)
                features = model_info.signature.inputs.input_names()
                group_dict[f"horizon_{i}"] = features
            feature_dict[group_name] = group_dict
        return feature_dict

    @property
    def feature_importance(self):
        importance_dict = {}
        for group_name, run_id in self.group_run_ids.items():
            group_dict = {}
            for i in range(self.n_horizon):
                path = mlflow.artifacts.download_artifacts(
                    run_id=run_id, artifact_path="feature_importance"
                )
                graph = plotly.io.read_json(os.path.join(path, f"horizon_{i}.json"))
                data = graph.data[0]
                group_dict[f"horizon_{i}"] = list(zip(data.y, data.x))
            importance_dict[group_name] = group_dict
        return importance_dict

    @property
    def feature_importance_graphs(self):
        graph_dict = {}
        for group_name, run_id in self.group_run_ids.items():
            group_dict = {}
            for i in range(self.n_horizon):
                path = mlflow.artifacts.download_artifacts(
                    run_id=run_id, artifact_path="feature_importance"
                )
                graph = plotly.io.read_json(os.path.join(path, f"horizon_{i}.json"))
                group_dict[f"horizon_{i}"] = graph
            graph_dict[group_name] = group_dict
        return graph_dict

    @property
    def cv_forecast(self):
        forecast_list = []
        for run_id in self.group_run_ids.values():
            path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="cv_forecast"
            )
            spark = SparkSession.builder.getOrCreate()
            cv_forecast = spark.read.parquet(path)
            forecast_list.append(cv_forecast)
        return functools.reduce(DataFrame.unionByName, forecast_list)

    @property
    def cv_forecast_graph(self):
        graph_dict = {}
        for group_name, run_id in self.group_run_ids.items():
            path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="cv_forecast_graph"
            )
            graph = plotly.io.read_json(os.path.join(path, f"cv_forecast_graph.json"))
            graph_dict[group_name] = graph
        return graph_dict

    def _filter_horizon(self, df, forecast_horizon):
        dates = df[self.date_col].sort_values().unique()
        forecast_dates = dates[[fh - 1 for fh in forecast_horizon]]
        return df[df[self.date_col].isin(forecast_dates)]

    def _filter_features(self, df, forecast_horizon):
        numeric_cols = [
            col
            for col in df.select_dtypes("number").columns
            if col
            not in [*self.id_cols, self.group_col, self.date_col, self.target_col]
        ]
        lag_cols = [
            col
            for col in numeric_cols
            if re.search("(^|_)lag(_|$)", col, re.IGNORECASE)
        ]
        keep_lags = (
            "("
            + ")|(".join(
                [
                    f"(^|_)lag_{i}(_|$)"
                    for i in range(
                        max(forecast_horizon),
                        max(forecast_horizon) + self.lag_feature_range + 1,
                    )
                ]
            )
            + ")"
        )
        return [
            col
            for col in numeric_cols
            if (
                (re.search(keep_lags, col, re.IGNORECASE) is not None)
                | (col not in lag_cols)
            )
        ]

    def _forecast_horizon(self, horizon_id):
        return list(
            range(
                horizon_id * self.model_horizon + 1,
                (horizon_id + 1) * self.model_horizon + 1,
            )
        )

    def _create_runs(self, df):
        with mlflow.start_run() as parent_run:
            self.run_id = parent_run.info.run_id
            mlflow.pyfunc.log_model(python_model=self, artifact_path="meta_model")
            group_names = [
                group[0]
                for group in df.select(self.group_col).dropDuplicates().collect()
            ]
            self.group_run_ids = {}
            for group_name in group_names:
                with mlflow.start_run(run_name=group_name, nested=True) as child_run:
                    self.group_run_ids[group_name] = child_run.info.run_id
            return self.group_run_ids

    def train(self, df):
        id_cols = self.id_cols
        date_col = self.date_col
        target_col = self.target_col
        max_forecast_horizon = self.max_forecast_horizon
        n_horizon = self.n_horizon
        date_frequency = self.date_frequency
        scoring = self.scoring
        max_hyperparam_evals = self.max_hyperparam_evals
        n_cv_splits = self.n_cv_splits
        hyperparam_space_fn = self.hyperparam_space_fn
        group_col = self.group_col
        tracking_uri = self.tracking_uri
        cv_step_length = self.cv_step_length
        n_jobs = self.n_jobs
        group_run_ids = self._create_runs(df)

        @F.pandas_udf(
            "group_name string, horizon_id int, status string",
            functionType=F.PandasUDFType.GROUPED_MAP,
        )
        def train_udf(df):
            mlflow.set_tracking_uri(tracking_uri)
            group_name = df[group_col].iloc[0]
            group_run_id = group_run_ids[group_name]
            horizon_id = df["horizon_id"].iloc[0]
            df = df.drop("horizon_id", axis=1)

            with mlflow.start_run(run_id=group_run_id):

                forecast_horizon = self._forecast_horizon(horizon_id)
                features = self._filter_features(df, forecast_horizon)

                optimizer = Optimizer(
                    id_cols=id_cols,
                    date_col=date_col,
                    target_col=target_col,
                    features=features,
                    cv_step_length=cv_step_length,
                    max_forecast_horizon=max_forecast_horizon,
                    forecast_horizon=forecast_horizon,
                    hyperparam_space_fn=hyperparam_space_fn,
                    date_frequency=date_frequency,
                    n_cv_splits=n_cv_splits,
                    scoring=scoring,
                    max_hyperparam_evals=max_hyperparam_evals,
                    n_jobs=n_jobs,
                )
                optimizer.run(df)

            return pd.DataFrame(
                [{"group_name": group_name, "horizon_id": horizon_id, "status": "ok"}]
            )

        (
            df.withColumn("date", F.to_timestamp("date"))
            .withColumn(
                "horizon_id", F.explode(F.array(list(map(F.lit, range(n_horizon)))))
            )
            .groupby(group_col, "horizon_id")
            .apply(train_udf)
            .collect()
        )
        evaluator = Evaluator(
            group_run_ids=group_run_ids,
            id_cols=id_cols,
            date_col=date_col,
            target_col=target_col,
            scoring=scoring,
        )
        evaluator.log_metric()
        evaluator.log_forecast_graph()

    def predict(self, context, model_input):
        tracking_uri = self.tracking_uri
        group_col = self.group_col
        parent_run_id = self.run_id
        id_cols = self.id_cols
        date_col = self.date_col
        n_horizon = self.n_horizon
        schema = ", ".join(
            [
                *[f"`{col}` string" for col in self.id_cols],
                f"`{date_col}` date",
                "prediction double",
            ]
        )

        @F.pandas_udf(
            schema,
            functionType=F.PandasUDFType.GROUPED_MAP,
        )
        def predict_udf(df):
            mlflow.set_tracking_uri(tracking_uri)
            group_name = df[group_col].iloc[0]
            filter_string = (
                f"tags.mlflow.parentRunId = '{parent_run_id}'"
                f" and tags.mlflow.runName = '{group_name}'"
            )
            run_id = mlflow.search_runs(filter_string=filter_string)["run_id"].iloc[0]
            horizon_id = df["horizon_id"].iloc[0]
            model_uri = f"runs:/{run_id}/models/horizon_{horizon_id}"

            model = mlflow.lightgbm.load_model(model_uri)
            model_info = mlflow.models.get_model_info(model_uri)
            forecast_horizon = self._forecast_horizon(horizon_id)
            features = model_info.signature.inputs.input_names()

            df = self._filter_horizon(df, forecast_horizon)
            df["prediction"] = model.predict(df[features])

            return df[[*id_cols, date_col, "prediction"]]

        return (
            model_input.withColumn(
                "horizon_id", F.explode(F.array(list(map(F.lit, range(n_horizon)))))
            )
            .groupBy(group_col, "horizon_id")
            .apply(predict_udf)
        )
