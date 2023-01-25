import pyspark.sql.functions as F

def load_wallmart():
    df = spark.read.parquet("./wallmart")
    df_train = df.filter(F.col("date") < "2016-01-01")
    df_test = df.filter(F.col("date") >= "2016-01-01")
    return df_train, df_test

