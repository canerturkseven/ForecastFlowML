import pytest
import datetime
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from forecastflowml import FeatureExtractor
from forecastflowml import ForecastFlowML
from lightgbm import LGBMRegressor


@pytest.fixture(scope="module")
def spark():
    spark = (
        SparkSession.builder.master("local[1]")
        .config("spark.sql.shuffle.partitions", 1)
        .getOrCreate()
    )
    yield spark
    spark.stop()


@pytest.fixture(scope="module")
def df(spark):
    df = spark.createDataFrame(
        data=[
            ("0", "0", "0", datetime.date(2023, 1, 1), 5, 3),
            ("0", "0", "0", datetime.date(2023, 1, 2), 2, 5),
            ("0", "0", "0", datetime.date(2023, 1, 3), 0, 3),
            ("0", "0", "0", datetime.date(2023, 1, 4), 4, 6),
            ("0" "0", "0", datetime.date(2023, 1, 5), 3, 2),
            ("0", "0", "0", datetime.date(2023, 1, 6), 3, 1),
            ("0", "1", "1", datetime.date(2023, 1, 1), 7, 1),
            ("0", "1", "1", datetime.date(2023, 1, 2), 8, 3),
            ("0", "1", "1", datetime.date(2023, 1, 3), 0, 5),
            ("0", "1", "1", datetime.date(2023, 1, 4), 0, 7),
            ("0", "1", "1", datetime.date(2023, 1, 5), 2, 2),
            ("0", "1", "1", datetime.date(2023, 1, 6), 2, 2),
        ],
        schema=["group_1", "group_2", "id", "date", "numeric_feature", "target"],
    )
    df_train = df.filter(F.col("date") < "2023-01-05").cache()
    df_test = df.filter(F.col("date") >= "2023-01-05").cache()
    df_train.count()
    df_test.count()
    return df_train, df_test


@pytest.mark.parametrize(
    "model_horizon, max_forecast_horizon",
    [([1], [1]), ([1], [1, 2]), ()],
)
def test_horizon(df):
    forecast_flow = ForecastFlowML(
        group_col="id",
        id_col="id",
        date_col="date",
        target_col="target",
        date_frequency="days",
        model_horizon=1,
        max_forecast_horizon=1,
        model=LGBMRegressor(),
    )
    forecast_flow.train(df)
    pass


def test_predict():
    pass


def test_cross_validate():
    pass


def test_grid_search():
    pass
