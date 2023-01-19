import mlflow
import optuna
import re
from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
)
from lightgbm import LGBMRegressor
from time_based_split import TimeBasedSplit
from artifacts import ModelArtifact, CrossValidationArtifact
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)


class Optimizer:
    def __init__(
        self,
        id_cols,
        date_col,
        target_col,
        max_forecast_horizon,
        model_horizon,
        n_cv_splits,
        hyperparam_space_fn,
        date_frequency,
        metric,
        max_hyperparam_evals,
    ):
        self.id_cols = id_cols
        self.date_col = date_col
        self.target_col = target_col
        self.max_forecast_horizon = max_forecast_horizon
        self.model_horizon = model_horizon
        self.hyperparam_space_fn = hyperparam_space_fn
        self.date_frequency = date_frequency
        self.metric = metric
        self.n_cv_splits = n_cv_splits
        self.max_hyperparam_evals = max_hyperparam_evals

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

    @staticmethod
    def _log_optimization_study(study):
        for func, output_file in [
            (plot_optimization_history, "optimize_history.html"),
            (plot_parallel_coordinate, "parallel_coordinate.html"),
            (plot_param_importances, "param_importances.html"),
        ]:
            func(study).write_html(output_file)
            mlflow.log_artifact(output_file)

    def _best_trial_callback(self, study, trial):

        if study.best_trial.number == trial.number:

            model_artifacts = trial.user_attrs.get("model_artifacts")
            for model_artifact in model_artifacts:
                model_artifact.log_model()
                model_artifact.log_feature_importance()

        cv_artifact = trial.user_attrs.get("cv_artifact")
        cv_artifact.log_forecast()
        cv_artifact.log_forecast_graph()
        cv_artifact.log_cv_metrics()

        mlflow.log_metric(self.metric, trial.value)

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

    def _objective_fn(self, trial, df):
        model_artifacts = []
        n_models = self.max_forecast_horizon // self.model_horizon

        for i in range(n_models):
            model = LGBMRegressor(**self.hyperparam_space_fn(trial))
            forecast_horizon = list(
                range(i * self.model_horizon + 1, (i + 1) * self.model_horizon + 1)
            )
            features = self._filter_features(df=df, forecast_horizon=forecast_horizon)
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

            model_artifact = ModelArtifact(
                model_name=f"fh_{i}",
                model=model,
                forecast_horizon=forecast_horizon,
                model_input_example=df[features].head(1),
                forecast=model_forecast,
            )
            model_artifacts.append(model_artifact)

        forecast = pd.concat(
            model_artifact.forecast for model_artifact in model_artifacts
        )
        cv_metrics = self._evaluate(forecast)
        cv_artifact = CrossValidationArtifact(
            cv_metrics=cv_metrics,
            forecast=forecast,
            id_cols=self.id_cols,
            df_train=df,
            date_col=self.date_col,
            target_col=self.target_col,
        )

        trial.set_user_attr("cv_artifact", cv_artifact)
        trial.set_user_attr("model_artifacts", model_artifacts)

        return cv_metrics[self.metric].mean()

    def run(self, df):
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: self._objective_fn(trial, df),
            n_trials=self.max_hyperparam_evals,
            callbacks=[self._best_trial_callback],
        )
        self._log_optimization_study(study)
