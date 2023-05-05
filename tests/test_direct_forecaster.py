import pytest
import pandas as pd
from datetime import date
from forecastflowml import FeatureExtractor
from forecastflowml.direct_forecaster import _DirectForecaster
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression


@pytest.fixture(scope="module")
def df(spark):
    df = pd.DataFrame(
        data=[
            ("0", "0", date(2023, 1, 1), 5, "a", 3),
            ("0", "0", date(2023, 1, 2), 2, "a", 5),
            ("0", "0", date(2023, 1, 3), 0, "a", 3),
            ("0", "0", date(2023, 1, 4), 4, "a", 6),
            ("0", "0", date(2023, 1, 5), 3, "a", 2),
            ("0", "0", date(2023, 1, 6), 3, "a", 1),
            ("0", "1", date(2023, 1, 1), 7, "b", 1),
            ("0", "1", date(2023, 1, 2), 8, "b", 3),
            ("0", "1", date(2023, 1, 3), 0, "b", 5),
            ("0", "1", date(2023, 1, 4), 0, "b", 7),
            ("0", "1", date(2023, 1, 5), 2, "b", 2),
            ("0", "1", date(2023, 1, 6), 2, "b", 2),
        ],
        columns=[
            "group",
            "id",
            "date",
            "numeric",
            "categorical",
            "target",
        ],
    )
    df["date"] = pd.to_datetime(df["date"])
    feature_extractor = FeatureExtractor(
        id_col="id",
        date_col="date",
        target_col="target",
        lag_window_features={
            "lag": [1, 2],
            "mean": [[window, lag] for window in [3] for lag in [1, 2]],
        },
        count_consecutive_values={"value": 0, "lags": [1, 2]},
    )
    df = feature_extractor.transform(df, spark=spark)
    df_train = df[df["date"] < "2023-01-05"]
    df_test = df[df["date"] >= "2023-01-05"]
    return df_train, df_test


def test_convert_categorical(df):
    df_train = df[0].copy()
    forecaster = _DirectForecaster(
        id_col="id",
        group_col="group",
        date_col="date",
        target_col="target",
        model=LGBMRegressor(),
        model_horizon=1,
        max_forecast_horizon=1,
        categorical_cols=["categorical"],
    )
    result = forecaster._convert_categorical(df_train)

    assert result["categorical"].dtype.name == "category"


@pytest.mark.parametrize(
    "model_horizon,max_forecast_horizon,forecast_horizon, expected_dates",
    [
        (1, 2, [1], ["2023-01-05"]),
        (1, 2, [2], ["2023-01-06"]),
        (2, 2, [1, 2], ["2023-01-05", "2023-01-06"]),
    ],
)
def test_filter_horizon(
    df, model_horizon, max_forecast_horizon, forecast_horizon, expected_dates
):
    df_test = df[1].copy()
    forecaster = _DirectForecaster(
        id_col="id",
        group_col="group",
        date_col="date",
        target_col="target",
        model=LGBMRegressor(),
        model_horizon=model_horizon,
        max_forecast_horizon=max_forecast_horizon,
    )
    result_df = forecaster._filter_horizon(df_test, forecast_horizon)
    result_dates = list(result_df["date"].astype(str).unique())
    assert result_dates == expected_dates


@pytest.mark.parametrize(
    "model_horizon, max_forecast_horizon, n_horizon, expected_forecast_horizon",
    [
        (3, 3, 0, (1, 2, 3)),
        (1, 3, 0, (1,)),
        (1, 3, 1, (2,)),
        (1, 3, 2, (3,)),
        (2, 4, 0, (1, 2)),
        (2, 4, 1, (3, 4)),
    ],
)
def test_forecast_horizon(
    model_horizon, max_forecast_horizon, n_horizon, expected_forecast_horizon
):
    forecaster = _DirectForecaster(
        id_col="id",
        group_col="group",
        date_col="date",
        target_col="target",
        model=LGBMRegressor(),
        model_horizon=model_horizon,
        max_forecast_horizon=max_forecast_horizon,
    )
    result_forecast_horizon = forecaster._forecast_horizon(n_horizon)
    assert result_forecast_horizon == expected_forecast_horizon


@pytest.mark.parametrize(
    "model_horizon, max_forecast_horizon, expected_model_horizon",
    [
        (1, 1, [(1,)]),
        (1, 2, [(1,), (2,)]),
        (2, 2, [(1, 2)]),
    ],
)
def test_fit_horizon(df, model_horizon, max_forecast_horizon, expected_model_horizon):
    df_train = df[0]
    forecaster = _DirectForecaster(
        id_col="id",
        group_col="group",
        date_col="date",
        target_col="target",
        model=LGBMRegressor(),
        model_horizon=model_horizon,
        max_forecast_horizon=max_forecast_horizon,
    )
    forecaster.fit(df_train)

    assert list(forecaster.model_.keys()) == expected_model_horizon
    assert len(forecaster.model_.keys()) == len(expected_model_horizon)


@pytest.mark.parametrize(
    "model_horizon, max_forecast_horizon, categorical_cols, use_lag_range, expected_features",
    [
        (
            1,
            1,
            ["categorical"],
            0,
            [
                [
                    "numeric",
                    "categorical",
                    "lag_1",
                    "window_3_lag_1_mean",
                    "count_consecutive_value_lag_1",
                ]
            ],
        ),
        (
            1,
            1,
            [],
            0,
            [
                [
                    "numeric",
                    "lag_1",
                    "window_3_lag_1_mean",
                    "count_consecutive_value_lag_1",
                ]
            ],
        ),
        (
            1,
            2,
            [],
            0,
            [
                [
                    "numeric",
                    "lag_1",
                    "window_3_lag_1_mean",
                    "count_consecutive_value_lag_1",
                ],
                [
                    "numeric",
                    "lag_2",
                    "window_3_lag_2_mean",
                    "count_consecutive_value_lag_2",
                ],
            ],
        ),
        (
            1,
            2,
            [],
            1,
            [
                [
                    "numeric",
                    "lag_1",
                    "lag_2",
                    "window_3_lag_1_mean",
                    "count_consecutive_value_lag_1",
                ],
                [
                    "numeric",
                    "lag_2",
                    "window_3_lag_2_mean",
                    "count_consecutive_value_lag_2",
                ],
            ],
        ),
        (
            2,
            2,
            [],
            0,
            [
                [
                    "numeric",
                    "lag_2",
                    "window_3_lag_2_mean",
                    "count_consecutive_value_lag_2",
                ]
            ],
        ),
    ],
)
def test_fit_features(
    df,
    model_horizon,
    max_forecast_horizon,
    categorical_cols,
    use_lag_range,
    expected_features,
):
    df_train = df[0]
    forecaster = _DirectForecaster(
        id_col="id",
        group_col="group",
        date_col="date",
        target_col="target",
        model=LGBMRegressor(),
        model_horizon=model_horizon,
        max_forecast_horizon=max_forecast_horizon,
        categorical_cols=categorical_cols,
        use_lag_range=use_lag_range,
    )
    forecaster.fit(df_train)
    result_features = [
        sorted(model.feature_name_) for model in forecaster.model_.values()
    ]
    expected_features = [sorted(features) for features in expected_features]
    assert result_features == expected_features


@pytest.mark.parametrize(
    "model_horizon, max_forecast_horizon,", [(1, 1), (1, 2), (2, 2)]
)
def test_predict(df, model_horizon, max_forecast_horizon):
    df_train = df[0]
    df_test = df[1]

    forecaster = _DirectForecaster(
        id_col="id",
        group_col="group",
        date_col="date",
        target_col="target",
        model=LGBMRegressor(),
        model_horizon=model_horizon,
        max_forecast_horizon=max_forecast_horizon,
    )
    forecaster.fit(df_train)
    predictions = forecaster.predict(df_test)

    expected_n_rows = len(df_test["id"].unique()) * max_forecast_horizon
    result_n_rows = len(predictions)

    assert result_n_rows == expected_n_rows


@pytest.mark.parametrize("model", [LGBMRegressor(), XGBRegressor(), LinearRegression()])
def test_lgb_xgb_sklearn(df, model):
    df_train = df[0]
    df_test = df[1]

    if isinstance(model, LinearRegression):
        df_train = df_train.dropna()
        df_test = df_test.dropna()

    forecaster = _DirectForecaster(
        id_col="id",
        group_col="group",
        date_col="date",
        target_col="target",
        model=model,
        model_horizon=1,
        max_forecast_horizon=1,
    )
    forecaster.fit(df_train)
    predictions = forecaster.predict(df_test)

    expected_n_rows = len(df_test["id"].unique()) * 1
    result_n_rows = len(predictions)

    assert result_n_rows == expected_n_rows


@pytest.mark.parametrize(
    "model_horizon, max_forecast_horizon, model",
    [
        (1, 1, LGBMRegressor(n_estimators=10)),
        (1, 2, {"0": LGBMRegressor(n_estimators=20)}),
        (2, 2, {"0": LGBMRegressor(n_estimators=30)}),
    ],
)
def test_fit_model_dict(df, model_horizon, max_forecast_horizon, model):
    df_train = df[0]
    group = df_train["group"].iloc[0]

    forecaster = _DirectForecaster(
        id_col="id",
        group_col="group",
        date_col="date",
        target_col="target",
        model=model,
        model_horizon=model_horizon,
        max_forecast_horizon=max_forecast_horizon,
    )
    forecaster.fit(df_train)

    model = model[group] if isinstance(model, dict) else model

    assert all(
        [
            model.get_params() == model_.get_params()
            for model_ in forecaster.model_.values()
        ]
    )
