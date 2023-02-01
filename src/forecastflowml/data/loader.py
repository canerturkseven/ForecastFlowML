import os
import pyspark.sql.functions as F


def load_walmart_m5(spark):
    path = os.path.join(os.path.dirname(__file__), "walmart_m5")
    df = spark.read.parquet(path)
    df_train = df.filter(F.col("date") <= "2016-05-22")
    df_test = df.filter(F.col("date") > "2016-05-22")
    return df_train, df_test
