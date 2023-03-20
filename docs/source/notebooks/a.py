#%%
import pyspark.sql.functions as f
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder.master("local[5]")
    .config("spark.driver.memory", "16g")
    .config("spark.sql.execution.arrow.enabled", "true")
    .config("spark.sql.adaptive.enabled", "false")
    .config("spark.sql.shuffle.partitions", "5")
    .getOrCreate()
)

df = (
    spark.range(1000)
    .select(
        f.col("id").alias("record_id"),
        (f.col("id") % 2).alias("device_id").cast("string"),
    )
    .withColumn("feature_1", f.rand() * 1)
    .withColumn("feature_2", f.rand() * 2)
    .withColumn("feature_3", f.rand() * 3)
    .withColumn(
        "label",
        (f.col("feature_1") + f.col("feature_2") + f.col("feature_3")) + f.rand(),
    )
)

#%%
import pyspark.sql.types as t

trainReturnSchema = t.StructType(
    [
        t.StructField("device_id", t.StringType()),  # unique device ID
        t.StructField(
            "serialized_model", t.StringType()
        ),  # path to the model for a given device
        # t.StructField("serialized_model", t.BinaryType()),  # path to the model for a given device
    ]
)

import pandas as pd
from lightgbm import LGBMRegressor

# from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import pickle


@f.pandas_udf(trainReturnSchema, functionType=f.PandasUDFType.GROUPED_MAP)
def train_model(df_pandas):
    """
    Trains an sklearn model on grouped instances
    """
    import os

    os.environ["ARROW_PRE_0_15_IPC_FORMAT"] = "1"

    # Pull metadata
    device_id = df_pandas["device_id"].iloc[0]

    # Train the model
    X = df_pandas[["feature_1", "feature_2", "feature_3"]]
    y = df_pandas["label"]
    model = LGBMRegressor()
    model.fit(X, y)

    serialized_model = str(pickle.dumps(model), "latin1")
    # serialized_model = pickle.dumps(model)

    returnDF = pd.DataFrame(
        [[device_id, serialized_model]],
        columns=["device_id", "serialized_model"],
    )

    return returnDF


df_model = df.groupby("device_id").apply(train_model)
#%%

applyReturnSchema = t.StructType(
    [
        t.StructField("device_id", t.StringType()),
        t.StructField("X", t.StringType()),
        t.StructField("y", t.StringType()),
    ]
)


@f.pandas_udf(applyReturnSchema, functionType=f.PandasUDFType.GROUPED_MAP)
def middle_convert(df):
    device_id = df["device_id"].iloc[0]
    X = str(pickle.dumps(df[["feature_1", "feature_2", "feature_3"]]), "latin1")
    y = str(pickle.dumps(df[["label"]]), "latin1")
    return pd.DataFrame([[device_id, X, y]], columns=["device_id", "X", "y"])


a = df.groupby("device_id").apply(middle_convert).toPandas()
b = df.toPandas()


#%%
applyReturnSchema = t.StructType(
    [
        t.StructField("record_id", t.IntegerType()),
        t.StructField("prediction", t.FloatType()),
    ]
)


@f.pandas_udf(applyReturnSchema, functionType=f.PandasUDFType.GROUPED_MAP)
def apply_model(df_pandas):
    """
    Applies model to data for a particular device, represented as a pandas DataFrame
    """
    import os

    os.environ["ARROW_PRE_0_15_IPC_FORMAT"] = "1"

    # Pull metadata
    device_id = df_pandas["device_id"].iloc[0]

    input_columns = ["feature_1", "feature_2", "feature_3"]
    X = df_pandas[input_columns]
    model = model_dict[device_id]
    prediction = model.predict(X)

    returnDF = pd.DataFrame(
        {"record_id": df_pandas["record_id"], "prediction": prediction}
    )
    return returnDF


df.groupby("device_id").apply(apply_model).collect()
# %%
