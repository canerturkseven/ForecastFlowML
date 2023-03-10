import os
import pandas as pd
import pyspark.sql.functions as F


def load_walmart_m5(spark):
    path = os.path.join(os.path.dirname(__file__), "walmart_m5")
    return spark.createDataFrame(pd.read_parquet(path))
