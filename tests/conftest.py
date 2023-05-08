import os
import sys
import pytest
import pyspark
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    os.environ["PYSPARK_PYTHON"] = sys.executable
    # spark_home = (
    #     os.environ.get("SPARK_HOME")
    #     if "SPARK_HOME" in os.environ
    #     else os.path.dirname(pyspark.__file__)
    # )
    # os.environ["SPARK_HOME"] = spark_home

    # if pyspark.__version__ < "3.1":
    #     os.environ[
    #         "SPARK_SUBMIT_OPTS"
    #     ] = "--illegal-access=permit -Dio.netty.tryReflectionSetAccessible=true "

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
