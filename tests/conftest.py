import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    spark = (
        SparkSession.builder.master("local[1]")
        .appName("pytest-spark-session")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.sql.execution.arrow.enabled", "false")
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
        .getOrCreate()
    )
    yield spark
    spark.stop()
