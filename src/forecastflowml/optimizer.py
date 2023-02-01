import optuna
from lightgbm import LGBMRegressor
from forecastflowml.time_based_split import TimeBasedSplit
from forecastflowml.artifacts import Artifact
import pandas as pd
from sklearn.model_selection import cross_val_score
from mlflow.models.signature import infer_signature


class Optimizer:
    def __init__(
        self,
        id_cols,
        date_col,
        target_col,
        features,
        forecast_horizon,
        max_forecast_horizon,
        n_cv_splits,
        hyperparam_space_fn,
        date_frequency,
        max_hyperparam_evals,
        cv_step_length,
        scoring,
        n_jobs=1,
    ):
        self.id_cols = id_cols
        self.date_col = date_col
        self.target_col = target_col
        self.features = features
        self.forecast_horizon = forecast_horizon
        self.max_forecast_horizon = max_forecast_horizon
        self.cv_step_length = cv_step_length
        self.hyperparam_space_fn = hyperparam_space_fn
        self.date_frequency = date_frequency
        self.n_cv_splits = n_cv_splits
        self.max_hyperparam_evals = max_hyperparam_evals
        self.n_jobs = n_jobs
        self.scoring = scoring

    def _objective_fn(self, trial, model, df, cv):
        hyperparams = self.hyperparam_space_fn(trial)
        model = model.set_params(**hyperparams)
        scores = cross_val_score(
            model,
            df[self.features],
            df[self.target_col],
            cv=cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
        )
        return scores.mean()

    def _cross_val_forecast(self, model, df, cv):
        forecast = []
        for i, fold in enumerate(cv):

            train_idx, test_idx = fold[0], fold[1]
            df_train, df_test = df.iloc[train_idx], df.copy().iloc[test_idx]

            model.fit(df_train[self.features], df_train[self.target_col])
            df_test["forecast"] = model.predict(df_test[self.features])
            df_test["cv"] = i
            forecast.append(
                df_test[
                    [*self.id_cols, self.date_col, "cv", self.target_col, "forecast"]
                ]
            )
        return pd.concat(forecast).reset_index(drop=True)

    def _reuse_best_trial(self, study, model, df, cv):
        model = model.set_params(**study.best_params)
        forecast = self._cross_val_forecast(model, df, cv)
        signature = infer_signature(df[self.features])
        model.fit(df[self.features], df[self.target_col])
        horizon_id = max(self.forecast_horizon) // len(self.forecast_horizon)

        return Artifact(
            horizon_id=horizon_id,
            model=model,
            signature=signature,
            forecast_horizon=self.forecast_horizon,
            study=study,
            forecast=forecast,
        )

    def run(self, df):
        model = LGBMRegressor()
        cv = TimeBasedSplit(
            date_col=self.date_col,
            date_frequency=self.date_frequency,
            n_splits=self.n_cv_splits,
            forecast_horizon=self.forecast_horizon,
            step_length=self.cv_step_length,
            end_offset=self.max_forecast_horizon - max(self.forecast_horizon),
        ).split(df)

        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: self._objective_fn(trial, model, df, cv),
            n_trials=self.max_hyperparam_evals,
        )

        artifact = self._reuse_best_trial(study, model, df, cv)
        artifact.log_model()
        artifact.log_feature_importance()
        artifact.log_forecast()
        artifact.log_optimization_visualisation()
