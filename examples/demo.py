#%%
# import packages
import mlflow
from forecastflowml.meta_model import MetaModel
from forecastflowml.data.loader import load_walmart_m5
from pyspark.sql import SparkSession

# create spark environment
spark = (
    SparkSession.builder.master("local[*]")
    .config("spark.driver.memory", "16g")
    .config("spark.sql.execution.arrow.enabled", "true")
    .config("spark.sql.adaptive.enabled", "false")
    .getOrCreate()
)

# load sample dataset
df_train, df_test = load_walmart_m5(spark)

# examine dataset columns
print(df_train.columns)

# define optuna hyperparmeter space
def hyperparam_space_fn(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.2, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 30, 40),
    }


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
    lag_feature_range=2,  #
    # cross validation and optimisation parameters
    n_cv_splits=1,  # number of time-based cv splits
    cv_step_length=28,  # number of dates between each cv folds
    max_hyperparam_evals=1,  # total number of optuna trials
    scoring="neg_mean_squared_error",  # sklearn scoring metric
    hyperparam_space_fn=hyperparam_space_fn,  # optuna hyperparameter space
    # mlflow parameters
    tracking_uri="./mlruns",  # Mlflow tracking URI
)


# launch mlflow server using command "mlflow ui" and train the model
# examine the training progress on mlflow platform
model.train(df_train)

# load meta model as mlflow.pyfunc
loaded_model = mlflow.pyfunc.load_model(f"runs:/{model.run_id}/meta_model")

# make predicttions, call an action such as collect or write
loaded_model.predict(df_test).write.parquet("forecast.parquet")

# %%
