import os
import pyspark.sql.functions as F


def load_walmart(spark):
    path = os.path.join(os.path.dirname(__file__), "walmart")
    df = spark.read.parquet(path).withColumn("date", F.to_timestamp("date"))
    df_train = df.filter(F.col("date") < "2016-01-01")
    df_test = df.filter(F.col("date") >= "2016-01-01")
    return df_train, df_test


