#%%
# import packages
import mlflow
from forecastflowml.meta_model import MetaModel
from forecastflowml.preprocessing import FeatureExtractor
from forecastflowml.data.loader import load_walmart_m5
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml import Pipeline

# create spark environment
spark = (
    SparkSession.builder.master("local[4]")
    .config("spark.driver.memory", "16g")
    .config("spark.sql.execution.arrow.enabled", "true")
    .config("spark.sql.adaptive.enabled", "false")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

# load sample dataset
df = load_walmart_m5(spark)
df.show()

# initialize feature extractor model
preprocessor = FeatureExtractor(
    id_col="id",
    date_col="date",
    date_frequency="day",
    target_col="sales",
    target_encodings=[
        {
            "partition_cols": ["item_id", "store_id"],
            "windows": [7, 14, 28],
            "lags": [7, 14, 21, 28],
            "functions": ["mean", "std"],
        },
        {
            "partition_cols": ["item_id", "store_id"],
            "windows": [1],
            "lags": [7, 8, 9, 14, 15, 16, 21, 22, 23, 28, 29, 30],
            "functions": ["mean"],
        },
    ],
    date_features=[
        "day_of_month",
        "day_of_week",
        "week_of_year",
        "quarter",
        "month",
        "year",
    ],
    history_lengths=["item_id", ["item_id", "store_id"]],
    encode_events={
        "cols": ["christmas"],
        "window": 15,
    },
    count_consecutive_values={"value": 0, "lags": [7, 14, 21, 28]},
)
# checkpoint dataframe to save intermediate results
df_preprocessed = preprocessor.transform(df).localCheckpoint()
df_preprocessed.show()

# split dataset into train and test
df_train = df_preprocessed.filter(F.col("date") <= "2016-05-22")
df_test = df_preprocessed.filter(F.col("date") > "2016-05-22")

# initialize meta model
model = MetaModel(
    # dataset parameters
    group_col="cat_id",  # column to slice dataframe
    id_cols=["id"],  # columns to use as time series identifier
    date_col="date",  # date column
    target_col="sales",  # target column
    date_frequency="days",  # date frequency (days, weeks, months, years) of dataset
    # model parameters
    model_horizon=7,  # horizon per model
    max_forecast_horizon=28,  # total forecast horizon
    lag_feature_range=2,  # extra lags to include as features based on model horizon
    # cross validation and optimisation parameters
    n_cv_splits=5,  # number of time-based cv splits
    cv_step_length=28,  # number of dates between each cv folds
    max_hyperparam_evals=1,  # total number of optuna trials
    scoring="neg_mean_squared_error",  # sklearn scoring metric
    # optuna hyperparameter space
    hyperparam_space_fn=lambda trial: {
        "num_leaves": trial.suggest_int("num_leaves", 20, 30)
    },
    # mlflow parameters
    tracking_uri="./mlruns",  # Mlflow tracking URI
)

# launch mlflow server using command "mlflow ui" and train the model
# examine the training progress on mlflow platform
model.train(df_train)

# load meta model as mlflow.pyfunc
loaded_model = mlflow.pyfunc.load_model(f"runs:/{model.run_id}/meta_model")

# make predictions
loaded_model.predict(df_test).show()

# %%
