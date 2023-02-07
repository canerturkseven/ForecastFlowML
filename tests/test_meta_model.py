from pyspark.sql import SparkSession

def test_dummy():
    assert 1 == 1

def test_spark_create():
    spark = SparkSession.builder.master("local[*]").getOrCreate()
    assert 1 == 1
