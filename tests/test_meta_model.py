import pytest
import mlflow
import datetime
import itertools
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from forecastflowml.meta_model import MetaModel
from dateutil.relativedelta import relativedelta


@pytest.fixture(scope="module")
def spark_session():
    session = (
        SparkSession.builder.master("local[*]")
        # .config("spark.sql.shuffle.partitions", 2)
        .getOrCreate()
    )
    yield session
    session.stop()


# @pytest.fixture(scope="module")
# def df_spark(spark_session):
#     return spark_session.createDataFrame(
#         [
#             ("0", "0", datetime.date(2023, 1, 1), 1.0, 1.0, 1.0),
#             ("1", "0", datetime.date(2023, 1, 1), 1.0, 1.0, 1.0),
#             ("2", "0", datetime.date(2023, 1, 1), 1.0, 1.0, 1.0),
#             ("3", "1", datetime.date(2023, 1, 1), 1.0, 1.0, 1.0),
#             ("4", "1", datetime.date(2023, 1, 1), 1.0, 1.0, 1.0),
#             ("5", "1", datetime.date(2023, 1, 1), 1.0, 1.0, 1.0),
#         ],
#         ["id", "group", "date", "lag_1", "lag_2", "target"],
#     ).cache()


# # @pytest.fixture(scope="module")
# # def df():
# #     return pd.DataFrame(
# #         {
# #             "id1": ["a1", "b1", "c1", "d1", "e1", "f1"],
# #             "id2": ["a2", "b2", "c2", "d2", "e2", "f2"],
# #             "group": [0, 0, 0, 1, 1, 1],
# #             "date": [
# #                 pd.Timestamp("2023-01-01"),
# #                 pd.Timestamp("2023-01-02"),
# #                 pd.Timestamp("2023-01-03"),
# #             ]
# #             * 2,
# #             "x_lag_1_x": [0] * 6,
# #             "x_lag_2_x": [0] * 6,
# #             "x_lag_3_x": [0] * 6,
# #             "x_lag_4_x": [0] * 6,
# #             "other_numeric": [0] * 6,
# #             "target": [0] * 6,
# #         }
# #     )


# @pytest.mark.parametrize(
#     "forecast_horizon, expected_index",
#     [
#         ([1], [0]),
#         ([2], [1]),
#         ([1, 2], [0, 1]),
#     ],
# )
# def test_filter_horizon(forecast_horizon, expected_index):
#     df = pd.DataFrame({"date": pd.date_range(start="2023-01-01", periods=2)})
#     meta_model = MetaModel(
#         id_cols="id",
#         group_col="group",
#         date_col="date",
#         target_col="target",
#         date_frequency="days",
#         max_forecast_horizon=3,
#         model_horizon=1,
#         hyperparam_space_fn=lambda trial: None,
#     )
#     filtered_df = meta_model._filter_horizon(df, forecast_horizon)
#     assert filtered_df.index.to_list() == expected_index


# @pytest.mark.parametrize(
#     "forecast_horizon, lag_feature_range, expected_features",
#     [
#         ([1], 0, ["x_lag_1_x", "other_numeric_col"]),
#         ([1, 2], 0, ["x_lag_2_x", "other_numeric_col"]),
#         ([1, 2, 3], 0, ["x_lag_3_x", "other_numeric_col"]),
#         ([2], 0, ["x_lag_2_x", "other_numeric_col"]),
#         ([3], 0, ["x_lag_3_x", "other_numeric_col"]),
#         ([2, 3], 0, ["x_lag_3_x", "other_numeric_col"]),
#         ([1], 1, ["x_lag_1_x", "x_lag_2_x", "other_numeric_col"]),
#         ([1], 2, ["x_lag_1_x", "x_lag_2_x", "x_lag_3_x", "other_numeric_col"]),
#         ([2], 1, ["x_lag_2_x", "x_lag_3_x", "other_numeric_col"]),
#     ],
# )
# def test_filter_features(forecast_horizon, lag_feature_range, expected_features):
#     df = pd.DataFrame(
#         [
#             {
#                 "string_col": "a",
#                 "x_lag_1_x": 0,
#                 "x_lag_2_x": 0,
#                 "x_lag_3_x": 0,
#                 "other_numeric_col": 0,
#             }
#         ]
#     )

#     meta_model = MetaModel(
#         id_cols="id",
#         group_col="group",
#         date_col="date",
#         target_col="target",
#         date_frequency="days",
#         max_forecast_horizon=1,
#         model_horizon=1,
#         hyperparam_space_fn=lambda trial: None,
#         lag_feature_range=lag_feature_range,
#     )

#     assert meta_model._filter_features(df, forecast_horizon) == expected_features


def _create_spark_df(
    spark_session,
    n_groups,
    date_freq,
    max_forecast_horizon,
    model_horizon,
    lag_feature_range,
):
    start_date = datetime.date(2023, 1, 1)

    lag_columns = [
        [f"lag_{i+j}" for j in range(lag_feature_range + 1)]
        for i in range(model_horizon, max_forecast_horizon + 1, model_horizon)
    ]
    unique_lag_columns = sorted(set([j for i in lag_columns for j in i]))
    numeric_columns = unique_lag_columns + ["other_numeric", "target"]
    all_columns = ["id", "group", "date"] + numeric_columns

    date_seq = [
        start_date + relativedelta(**{date_freq: i})
        for i in range(max_forecast_horizon + 1)
    ]
    ts_id = [
        i
        for i in range(
            n_groups * (max_forecast_horizon + 1),
        )
    ]
    group_date = itertools.product([i for i in range(n_groups)], date_seq)
    numeric_values = np.random.rand(len(ts_id), len(numeric_columns)).round(2).tolist()
    data = [[a, *b, *c] for a, b, c in zip(ts_id, group_date, numeric_values)]

    df = spark_session.createDataFrame(data, all_columns)
    df_train = df.filter(F.col("date") == start_date)
    df_test = df.filter(F.col("date") != start_date)

    return df_train, df_test


@pytest.mark.parametrize(
    "n_groups, date_freq, max_forecast_horizon, model_horizon,lag_feature_range",
    [
        (1, "days", 2, 1, 0),
    ],
)
def test_train_predict(
    spark_session,
    n_groups,
    date_freq,
    max_forecast_horizon,
    model_horizon,
    lag_feature_range,
):

    df_train, df_test = _create_spark_df(
        spark_session=spark_session,
        n_groups=n_groups,
        date_freq=date_freq,
        max_forecast_horizon=max_forecast_horizon,
        model_horizon=model_horizon,
        lag_feature_range=lag_feature_range,
    )
    df_test.show()
    model = MetaModel(
        id_cols=["id"],
        group_col="group",
        date_col="date",
        target_col="target",
        date_frequency=date_freq,
        max_forecast_horizon=max_forecast_horizon,
        model_horizon=model_horizon,
        lag_feature_range=lag_feature_range,
        hyperparam_space_fn=lambda trial: {
            "n_estimators": trial.suggest_float("n_estimators", 1, 1)
        },
    )

    model.train(df_train)

    loaded_model = mlflow.pyfunc.load_model(f"runs:/{model.run_id}/meta_model")

    assert loaded_model.predict(df_test).count() == df_test.count()
