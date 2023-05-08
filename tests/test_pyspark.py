def test_pyspark(spark):
    spark.range(10).show()
    assert True


def test_pandas_udf(spark):
    df = spark.range(10)

    def udf(df_pandas):
        return df_pandas

    assert (
        df.groupby("id").applyInPandas(udf, schema=df.schema).collect() == df.collect()
    )
