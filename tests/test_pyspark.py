import pyspark
import pyspark.sql.functions as F


def test_pyspark(spark):
    spark.range(10).show()
    assert True


def test_pandas_udf(spark):
    df = spark.range(10)

    def udf(df_pandas):
        return df_pandas

    if pyspark.__version__ < "3":
        pandas_udf = F.pandas_udf(
            udf,
            returnType=df.schema,
            functionType=F.PandasUDFType.GROUPED_MAP,
        )
        assert df.groupby("id").apply(pandas_udf).collect() == df.collect()
    else:
        assert (
            df.groupby("id").applyInPandas(udf, schema=df.schema).collect()
            == df.collect()
        )
