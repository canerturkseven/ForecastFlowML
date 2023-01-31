#%%
from pyspark.sql import SparkSession
from forecastmlflow.meta_model import MetaModel
from forecastmlflow.data.loader import load_walmart

spark = (
    SparkSession.builder.master("local[4]")
    .config("spark.driver.memory", "30g")
    .config("spark.sql.shuffle.partitions", 4)
    .config("spark.driver.maxResultSize", "10g")
    .config("spark.sql.execution.arrow.enabled", "true")
    .config("spark.executor.heartbeatInterval", "36000000s")
    .config("spark.sql.adaptive.enabled", "false")
    .config("spark.network.timeout", "72000000s")
    .getOrCreate()
)

df_train, df_test = load_walmart(spark)


def hyperparam_space_fn(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
    }


model = MetaModel(
    group_col="cat_id",
    id_cols=["id"],
    date_col="date",
    date_frequency="days",
    n_cv_splits=5,
    model_horizon=7,
    max_forecast_horizon=7 * 4,
    target_col="sales",
    tracking_uri="./mlruns",
    max_hyperparam_evals=1,
    scoring="neg_mean_squared_error",
    hyperparam_space_fn=hyperparam_space_fn,
)
model.train(df_train)
#%%
