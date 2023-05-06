import os
import pytest
import pyspark
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():

    if ((pyspark.__version__ < "3.1") & (pyspark.__version__ >= "3.0")):
        os.environ[
            "SPARK_SUBMIT_OPTS"
        ] = "--illegal-access=permit -Dio.netty.tryReflectionSetAccessible=true "

    spark = (
        SparkSession.builder.master("local[1]")
        .appName("pytest-spark-session")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.sql.execution.pyarrow.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
        .getOrCreate()
    )
    yield spark
    spark.stop()
