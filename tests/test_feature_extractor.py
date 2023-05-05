import pytest
import pandas as pd
from datetime import date
from forecastflowml import FeatureExtractor


@pytest.fixture(scope="module")
def df_input(spark):
    df = spark.createDataFrame(
        data=[
            ("0", date(2023, 1, 1), 0.0),
            ("0", date(2023, 1, 2), 0.0),
            ("0", date(2023, 1, 3), 5.0),
            ("1", date(2023, 1, 1), 7.0),
            ("1", date(2023, 1, 2), 0.0),
            ("1", date(2023, 1, 3), 2.0),
        ],
        schema=["id", "date", "target"],
    ).cache()
    df.count()
    return df


@pytest.mark.parametrize(
    "lag_list, expected_data, expected_schema",
    [
        (
            [1],
            [
                ("0", date(2023, 1, 1), 0.0, None),
                ("0", date(2023, 1, 2), 0.0, 0.0),
                ("0", date(2023, 1, 3), 5.0, 0.0),
                ("1", date(2023, 1, 1), 7.0, None),
                ("1", date(2023, 1, 2), 0.0, 7.0),
                ("1", date(2023, 1, 3), 2.0, 0.0),
            ],
            ["id", "date", "target", "lag_1"],
        ),
        (
            [2],
            [
                ("0", date(2023, 1, 1), 0.0, None),
                ("0", date(2023, 1, 2), 0.0, None),
                ("0", date(2023, 1, 3), 5.0, 0.0),
                ("1", date(2023, 1, 1), 7.0, None),
                ("1", date(2023, 1, 2), 0.0, None),
                ("1", date(2023, 1, 3), 2.0, 7.0),
            ],
            ["id", "date", "target", "lag_2"],
        ),
    ],
)
def test_lag(df_input, spark, lag_list, expected_data, expected_schema):
    df_expected = spark.createDataFrame(expected_data, expected_schema)
    model = FeatureExtractor(
        id_col="id",
        date_col="date",
        target_col="target",
        lag_window_features={"lag": lag_list},
    )
    df_result = model.transform(df_input)
    assert df_result.collect() == df_expected.collect()


@pytest.mark.parametrize(
    "func, window_lag, expected_data, expected_schema",
    [
        (
            "mean",
            [[2, 0]],
            [
                ("0", date(2023, 1, 1), 0.0, 0.0),
                ("0", date(2023, 1, 2), 0.0, 0.0),
                ("0", date(2023, 1, 3), 5.0, 2.5),
                ("1", date(2023, 1, 1), 7.0, 7.0),
                ("1", date(2023, 1, 2), 0.0, 3.5),
                ("1", date(2023, 1, 3), 2.0, 1.0),
            ],
            ["id", "date", "target", "window_2_lag_0_mean"],
        ),
        (
            "sum",
            [[2, 0]],
            [
                ("0", date(2023, 1, 1), 0.0, 0.0),
                ("0", date(2023, 1, 2), 0.0, 0.0),
                ("0", date(2023, 1, 3), 5.0, 5.0),
                ("1", date(2023, 1, 1), 7.0, 7.0),
                ("1", date(2023, 1, 2), 0.0, 7.0),
                ("1", date(2023, 1, 3), 2.0, 2.0),
            ],
            ["id", "date", "target", "window_2_lag_0_sum"],
        ),
        (
            "mean",
            [[2, 1]],
            [
                ("0", date(2023, 1, 1), 0.0, None),
                ("0", date(2023, 1, 2), 0.0, 0.0),
                ("0", date(2023, 1, 3), 5.0, 0.0),
                ("1", date(2023, 1, 1), 7.0, None),
                ("1", date(2023, 1, 2), 0.0, 7.0),
                ("1", date(2023, 1, 3), 2.0, 3.5),
            ],
            ["id", "date", "target", "window_2_lag_1_mean"],
        ),
    ],
)
def test_lag_window(df_input, spark, func, window_lag, expected_data, expected_schema):
    df_expected = spark.createDataFrame(expected_data, expected_schema)
    model = FeatureExtractor(
        id_col="id",
        date_col="date",
        target_col="target",
        lag_window_features={func: window_lag},
    )
    df_result = model.transform(df_input)
    assert df_result.collect() == df_expected.collect()


def test_history_length(df_input, spark):
    expected_data = [
        ("0", date(2023, 1, 1), 0.0, 1),
        ("0", date(2023, 1, 2), 0.0, 2),
        ("0", date(2023, 1, 3), 5.0, 3),
        ("1", date(2023, 1, 1), 7.0, 1),
        ("1", date(2023, 1, 2), 0.0, 2),
        ("1", date(2023, 1, 3), 2.0, 3),
    ]
    expected_schema = ["id", "date", "target", "history_length"]
    df_expected = spark.createDataFrame(expected_data, expected_schema)
    model = FeatureExtractor(
        id_col="id",
        date_col="date",
        target_col="target",
        history_length=True,
    )
    df_result = model.transform(df_input)
    assert df_result.collect() == df_expected.collect()


@pytest.mark.parametrize(
    "lag_list, expected_data, expected_schema",
    [
        (
            [1],
            [
                ("0", date(2023, 1, 1), 0.0, None),
                ("0", date(2023, 1, 2), 0.0, 1),
                ("0", date(2023, 1, 3), 5.0, 2),
                ("1", date(2023, 1, 1), 7.0, None),
                ("1", date(2023, 1, 2), 0.0, 0),
                ("1", date(2023, 1, 3), 2.0, 1),
            ],
            [
                "id",
                "date",
                "target",
                "count_consecutive_values_lag_1",
            ],
        ),
        (
            [2],
            [
                ("0", date(2023, 1, 1), 0.0, None),
                ("0", date(2023, 1, 2), 0.0, None),
                ("0", date(2023, 1, 3), 5.0, 1),
                ("1", date(2023, 1, 1), 7.0, None),
                ("1", date(2023, 1, 2), 0.0, None),
                ("1", date(2023, 1, 3), 2.0, 0),
            ],
            [
                "id",
                "date",
                "target",
                "count_consecutive_values_lag_2",
            ],
        ),
    ],
)
def test_count_consecutive_values(
    df_input, spark, lag_list, expected_data, expected_schema
):
    df_expected = spark.createDataFrame(expected_data, expected_schema)
    model = FeatureExtractor(
        id_col="id",
        date_col="date",
        target_col="target",
        count_consecutive_values={"value": 0, "lags": lag_list},
    )
    df_result = model.transform(df_input)
    assert df_result.collect() == df_expected.collect()


@pytest.mark.parametrize(
    "feature, input_data, input_schema, expected_data, expected_schema",
    [
        (
            "day_of_week",
            [
                ("0", date(2023, 4, 10), 0),
                ("0", date(2023, 4, 15), 0),
                ("0", date(2023, 4, 16), 0),
            ],
            ["id", "date", "target"],
            [
                ("0", date(2023, 4, 10), 0, 2),
                ("0", date(2023, 4, 15), 0, 7),
                ("0", date(2023, 4, 16), 0, 1),
            ],
            ["id", "date", "target", "day_of_week"],
        ),
        (
            "day_of_year",
            [
                ("0", date(2023, 1, 1), 0),
                ("0", date(2023, 12, 31), 0),
            ],
            ["id", "date", "target"],
            [
                ("0", date(2023, 1, 1), 0, 1),
                ("0", date(2023, 12, 31), 0, 365),
            ],
            ["id", "date", "target", "day_of_year"],
        ),
        (
            "day_of_month",
            [
                ("0", date(2023, 1, 1), 0),
                ("0", date(2023, 1, 31), 0),
            ],
            ["id", "date", "target"],
            [
                ("0", date(2023, 1, 1), 0, 1),
                ("0", date(2023, 1, 31), 0, 31),
            ],
            ["id", "date", "target", "day_of_month"],
        ),
        (
            "week_of_year",
            [
                ("0", date(2023, 1, 2), 0),
                ("0", date(2023, 12, 25), 0),
            ],
            ["id", "date", "target"],
            [
                ("0", date(2023, 1, 2), 0, 1),
                ("0", date(2023, 12, 25), 0, 52),
            ],
            ["id", "date", "target", "day_of_month"],
        ),
        (
            "week_of_month",
            [
                ("0", date(2023, 1, 7), 0),
                ("0", date(2023, 12, 14), 0),
                ("0", date(2023, 12, 21), 0),
                ("0", date(2023, 12, 28), 0),
            ],
            ["id", "date", "target"],
            [
                ("0", date(2023, 1, 7), 0, 1),
                ("0", date(2023, 12, 14), 0, 2),
                ("0", date(2023, 12, 21), 0, 3),
                ("0", date(2023, 12, 28), 0, 4),
            ],
            ["id", "date", "target", "week_of_month"],
        ),
        (
            "weekend",
            [
                ("0", date(2023, 1, 2), 0),
                ("0", date(2023, 1, 6), 0),
                ("0", date(2023, 1, 7), 0),
                ("0", date(2023, 1, 8), 0),
            ],
            ["id", "date", "target"],
            [
                ("0", date(2023, 1, 2), 0, 0),
                ("0", date(2023, 1, 6), 0, 0),
                ("0", date(2023, 1, 7), 0, 1),
                ("0", date(2023, 1, 8), 0, 1),
            ],
            ["id", "date", "target", "weekend"],
        ),
        (
            "month",
            [
                ("0", date(2023, 1, 1), 0),
                ("0", date(2023, 12, 1), 0),
            ],
            ["id", "date", "target"],
            [
                ("0", date(2023, 1, 1), 0, 1),
                ("0", date(2023, 12, 1), 0, 12),
            ],
            ["id", "date", "target", "month"],
        ),
        (
            "quarter",
            [
                ("0", date(2023, 1, 1), 0),
                ("0", date(2023, 4, 1), 0),
                ("0", date(2023, 7, 1), 0),
                ("0", date(2023, 10, 1), 0),
            ],
            ["id", "date", "target"],
            [
                ("0", date(2023, 1, 1), 0, 1),
                ("0", date(2023, 4, 1), 0, 2),
                ("0", date(2023, 7, 1), 0, 3),
                ("0", date(2023, 10, 1), 0, 4),
            ],
            ["id", "date", "target", "quarter"],
        ),
        (
            "year",
            [
                ("0", date(2023, 1, 1), 0),
                ("0", date(2024, 1, 1), 0),
            ],
            ["id", "date", "target"],
            [
                ("0", date(2023, 1, 1), 0, 2023),
                ("0", date(2024, 1, 1), 0, 2024),
            ],
            ["id", "date", "target", "year"],
        ),
    ],
)
def test_date_features(
    spark,
    feature,
    input_data,
    input_schema,
    expected_data,
    expected_schema,
):
    df_input = spark.createDataFrame(input_data, input_schema)
    df_expected = spark.createDataFrame(expected_data, expected_schema)
    model = FeatureExtractor(
        id_col="id",
        date_col="date",
        target_col="target",
        date_features=[feature],
    )
    df_result = model.transform(df_input)
    assert df_result.collect() == df_expected.collect()


def test_pandas_dataframe(df_input, spark):
    model = FeatureExtractor(
        id_col="id",
        date_col="date",
        target_col="target",
        lag_window_features={"lag": [1]},
    )
    df_result = model.transform(df_input.toPandas(), spark=spark)
    assert isinstance(df_result, pd.DataFrame)
