import pytest
import pandas as pd
import numpy as np
from forecastflowml.direct_forecaster import _DirectForecaster
from forecastflowml.model_selection import (
    _TimeBasedSplit,
    _cross_val_predict,
    _score_func,
)
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error


@pytest.fixture(scope="module")
def df():
    n_periods = 10
    df = pd.DataFrame(
        {
            "group": "0",
            "id": "0",
            "feature": np.random.rand(n_periods),
            "target": np.random.rand(n_periods),
        }
    )
    df_daily = df.assign(date=pd.date_range("2023-01-01", freq="D", periods=n_periods))
    df_weekly = df.assign(
        date=pd.date_range("2023-01-01", freq="W-SUN", periods=n_periods)
    )
    df_monthly = df.assign(
        date=pd.date_range("2023-01-01", freq="MS", periods=n_periods)
    )

    return df_daily, df_weekly, df_monthly


@pytest.mark.parametrize(
    "forecast_horizon, n_splits, step_length, max_train_size, expected_index",
    [
        ([1], 1, None, None, [([0, 1, 2, 3, 4, 5, 6, 7, 8], [9])]),
        (
            [1],
            2,
            None,
            None,
            [([0, 1, 2, 3, 4, 5, 6, 7, 8], [9]), ([0, 1, 2, 3, 4, 5, 6, 7], [8])],
        ),
        (
            [1],
            2,
            2,
            None,
            [([0, 1, 2, 3, 4, 5, 6, 7, 8], [9]), ([0, 1, 2, 3, 4, 5, 6], [7])],
        ),
        (
            [1],
            1,
            None,
            3,
            [([6, 7, 8], [9])],
        ),
        (
            [1, 2],
            1,
            None,
            None,
            [([0, 1, 2, 3, 4, 5, 6, 7], [8, 9])],
        ),
        (
            [1, 2],
            2,
            None,
            None,
            [([0, 1, 2, 3, 4, 5, 6, 7], [8, 9]), ([0, 1, 2, 3, 4, 5], [6, 7])],
        ),
        (
            [1, 2],
            2,
            1,
            None,
            [([0, 1, 2, 3, 4, 5, 6, 7], [8, 9]), ([0, 1, 2, 3, 4, 5, 6], [7, 8])],
        ),
        ([1, 2], 1, None, 3, [([5, 6, 7], [8, 9])]),
    ],
)
def test_time_based_split(
    df, forecast_horizon, n_splits, step_length, max_train_size, expected_index
):
    df_daily, df_weekly, df_monthly = df[0], df[1], df[2]
    cv_daily_model = _TimeBasedSplit(
        date_frequency="days",
        date_col="date",
        forecast_horizon=forecast_horizon,
        max_train_size=max_train_size,
        n_splits=n_splits,
        step_length=step_length,
    )
    cv_weekly_model = _TimeBasedSplit(
        date_frequency="weeks",
        date_col="date",
        forecast_horizon=forecast_horizon,
        max_train_size=max_train_size,
        n_splits=n_splits,
        step_length=step_length,
    )
    cv_monthly_model = _TimeBasedSplit(
        date_frequency="months",
        date_col="date",
        forecast_horizon=forecast_horizon,
        max_train_size=max_train_size,
        n_splits=n_splits,
        step_length=step_length,
    )
    cv_daily = cv_daily_model.split(df_daily)
    cv_weekly = cv_weekly_model.split(df_weekly)
    cv_monthly = cv_monthly_model.split(df_monthly)
    assert cv_daily == expected_index
    assert cv_weekly == expected_index
    assert cv_monthly == expected_index


@pytest.mark.parametrize(
    "n_splits, model_horizon, max_forecast_horizon",
    [
        (1, 1, 1),
        (1, 1, 2),
        (2, 1, 1),
        (2, 1, 2),
    ],
)
def test_cross_val_predict(df, n_splits, model_horizon, max_forecast_horizon):
    df = df[0]
    n_splits = 2
    model_horizon = 1
    max_forecast_horizon = 2
    forecast_horizon = list(range(1, max_forecast_horizon + 1))
    cv_model = _TimeBasedSplit(
        date_frequency="days",
        date_col="date",
        forecast_horizon=forecast_horizon,
        n_splits=n_splits,
    )
    forecaster = _DirectForecaster(
        id_col="id",
        group_col="group",
        date_col="date",
        target_col="target",
        model=LGBMRegressor(),
        model_horizon=model_horizon,
        max_forecast_horizon=max_forecast_horizon,
    )
    cv_predictions = _cross_val_predict(
        forecaster=forecaster,
        df=df,
        cv=cv_model.split(df),
        refit=False,
    )
    assert list(cv_predictions["cv"].unique()) == list(map(str, range(n_splits)))
    assert (
        len(cv_predictions) == len(cv_predictions["cv"].unique()) * max_forecast_horizon
    )


def test_score_func():
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    assert mean_squared_error(y_true, y_pred) == -1 * _score_func(
        y_true, y_pred, "neg_mean_squared_error"
    )
    assert mean_squared_error(y_true, y_pred, squared=False) == -1 * _score_func(
        y_true, y_pred, "neg_root_mean_squared_error"
    )
