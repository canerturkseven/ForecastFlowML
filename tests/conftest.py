import os
import sys
import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    os.environ["PYSPARK_PYTHON"] = sys.executable

    spark = (
        SparkSession.builder.master("local[1]")
        .appName("pytest-spark-session")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.sql.execution.pyarrow.enabled", "true")
        .getOrCreate()
    )
    yield spark
    spark.stop()
