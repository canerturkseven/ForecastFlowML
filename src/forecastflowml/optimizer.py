import optuna
from lightgbm import LGBMRegressor
import pandas as pd
from sklearn.model_selection import cross_val_score
from forecastflowml.time_based_split import TimeBasedSplit
from forecastflowml.artifacts import (
    log_train_data,
    log_model,
    log_feature_importance,
    log_optimization_visualisation,
    log_cv_forecast,
)


class Optimizer:
    def __init__(
        self,
        group_name,
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
        scoring_metric,
        n_jobs=1,
    ):
        self.group_name = group_name
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
        self.scoring_metric = scoring_metric

    def _objective_fn(self, trial, model, df, cv):
        hyperparams = self.hyperparam_space_fn(trial)
        model = model.set_params(**hyperparams)
        scores = cross_val_score(
            model,
            df[self.features],
            df[self.target_col],
            cv=cv,
            scoring=self.scoring_metric,
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
            df_test["cv"] = str(i)
            forecast.append(
                df_test[
                    [*self.id_cols, self.date_col, "cv", self.target_col, "forecast"]
                ]
            )
        return pd.concat(forecast).reset_index(drop=True)

    def _log_best_trial(self, study, model, df, cv):
        group_name = self.group_name
        model = model.set_params(**study.best_params)
        cv_forecast = self._cross_val_forecast(model, df, cv)
        model.fit(df[self.features], df[self.target_col])
        horizon_id = (max(self.forecast_horizon) // len(self.forecast_horizon)) - 1
        log_train_data(group_name, df, horizon_id)
        log_model(df, model, horizon_id)
        log_feature_importance(group_name, model, horizon_id)
        log_cv_forecast(group_name, cv_forecast, horizon_id)
        log_optimization_visualisation(group_name, study, horizon_id)

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
        self._log_best_trial(study, model, df, cv)
