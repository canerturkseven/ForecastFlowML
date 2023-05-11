import os

def load_walmart_m5(spark):
    path = os.path.join(os.path.dirname(__file__), "walmart_m5")
    return spark.read.parquet(path)
