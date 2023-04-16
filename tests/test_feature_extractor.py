import pytest
import datetime
from pyspark.sql import SparkSession
from forecastflowml import FeatureExtractor


@pytest.fixture(scope="module")
def spark_session():
    session = (
        SparkSession.builder.master("local[1]")
        .config("spark.sql.shuffle.partitions", 1)
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture(scope="module")
def df_input(spark_session):
    df = spark_session.createDataFrame(
        data=[
            ("0", datetime.date(2023, 1, 1), 0),
            ("0", datetime.date(2023, 1, 2), 0),
            ("0", datetime.date(2023, 1, 3), 5),
            ("1", datetime.date(2023, 1, 1), 7),
            ("1", datetime.date(2023, 1, 2), 1),
            ("1", datetime.date(2023, 1, 3), 2),
        ],
        schema=["id", "date", "target"],
    ).cache()
    df.count()
    return df


@pytest.mark.parametrize(
    "lag_list, expected_data, schema",
    [
        (
            [1],
            [
                ("0", datetime.date(2023, 1, 1), 0, None),
                ("0", datetime.date(2023, 1, 2), 0, 0),
                ("0", datetime.date(2023, 1, 3), 5, 0),
                ("1", datetime.date(2023, 1, 1), 7, None),
                ("1", datetime.date(2023, 1, 2), 1, 7),
                ("1", datetime.date(2023, 1, 3), 2, 1),
            ],
            ["id", "date", "target", "lag_1"],
        ),
        (
            [1, 2],
            [
                ("0", datetime.date(2023, 1, 1), 0, None, None),
                ("0", datetime.date(2023, 1, 2), 0, 0, None),
                ("0", datetime.date(2023, 1, 3), 5, 0, 0),
                ("1", datetime.date(2023, 1, 1), 7, None, None),
                ("1", datetime.date(2023, 1, 2), 1, 7, None),
                ("1", datetime.date(2023, 1, 3), 2, 1, 7),
            ],
            ["id", "date", "target", "lag_1", "lag_2"],
        ),
    ],
)
def test_lag(df_input, spark_session, lag_list, expected_data, schema):
    df_expected = spark_session.createDataFrame(expected_data, schema)
    model = FeatureExtractor(
        id_col="id",
        date_col="date",
        target_col="target",
        lag_window_features={"lag": lag_list},
    )
    df_result = model.transform(df_input)
    assert df_result.collect() == df_expected.collect()
