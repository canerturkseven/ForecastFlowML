import pytest
import datetime
import pyspark.sql.functions as F
from forecastflowml import ForecastFlowML
from lightgbm import LGBMRegressor


@pytest.fixture(scope="module")
def df(spark):
    df = spark.createDataFrame(
        data=[
            ("0", "0", datetime.date(2023, 1, 1), 5, 3),
            ("0", "0", datetime.date(2023, 1, 2), 2, 5),
            ("0", "0", datetime.date(2023, 1, 3), 0, 3),
            ("0", "0", datetime.date(2023, 1, 4), 4, 6),
            ("0", "0", datetime.date(2023, 1, 5), 3, 2),
            ("0", "0", datetime.date(2023, 1, 6), 3, 1),
            ("1", "1", datetime.date(2023, 1, 1), 7, 1),
            ("1", "1", datetime.date(2023, 1, 2), 8, 3),
            ("1", "1", datetime.date(2023, 1, 3), 0, 5),
            ("1", "1", datetime.date(2023, 1, 4), 0, 7),
            ("1", "1", datetime.date(2023, 1, 5), 2, 2),
            ("1", "1", datetime.date(2023, 1, 6), 2, 2),
        ],
        schema=["group", "id", "date", "numeric_feature", "target"],
    )
    df_train = df.filter(F.col("date") < "2023-01-05").cache()
    df_test = df.filter(F.col("date") >= "2023-01-05").cache()
    df_train.count()
    df_test.count()
    return df_train, df_test


def test_train(df):
    df_train = df[0]
    n_group = df_train.select("group").dropDuplicates().count()
    forecast_flow = ForecastFlowML(
        group_col="group",
        id_col="id",
        date_col="date",
        target_col="target",
        date_frequency="days",
        model_horizon=1,
        max_forecast_horizon=2,
        model=LGBMRegressor(),
    )
    trained_models = forecast_flow.train(df_train)
    assert trained_models.count() == n_group
    assert len(trained_models.select("forecast_horizon").collect()[0][0]) == n_group
    assert len(trained_models.select("model").collect()[0][0]) == n_group


def test_train_local_result(df):
    df_train = df[0]
    n_group = df_train.select("group").dropDuplicates().count()
    forecast_flow = ForecastFlowML(
        group_col="group",
        id_col="id",
        date_col="date",
        target_col="target",
        date_frequency="days",
        model_horizon=1,
        max_forecast_horizon=2,
        model=LGBMRegressor(),
    )
    forecast_flow.train(df_train, local_result=True)
    assert len(forecast_flow.model_) == n_group
    assert len(forecast_flow.model_["forecast_horizon"].iloc[0]) == n_group
    assert len(forecast_flow.model_["model"].iloc[0]) == n_group


def test_train_pandas_dataframe(df, spark):
    df_train_pandas = df[0].toPandas()
    n_group = len(df_train_pandas["group"].unique())
    forecast_flow = ForecastFlowML(
        group_col="group",
        id_col="id",
        date_col="date",
        target_col="target",
        date_frequency="days",
        model_horizon=1,
        max_forecast_horizon=2,
        model=LGBMRegressor(),
    )
    forecast_flow.train(df_train_pandas, spark=spark)
    assert len(forecast_flow.model_) == n_group
    assert len(forecast_flow.model_["forecast_horizon"].iloc[0]) == n_group
    assert len(forecast_flow.model_["model"].iloc[0]) == n_group


def test_predict(df):
    df_train, df_test = df[0], df[1]
    forecast_flow = ForecastFlowML(
        group_col="group",
        id_col="id",
        date_col="date",
        target_col="target",
        date_frequency="days",
        model_horizon=1,
        max_forecast_horizon=2,
        model=LGBMRegressor(),
    )
    trained_models = forecast_flow.train(df_train)
    predictions = forecast_flow.predict(df_test, trained_models)
    assert predictions.count() == df_test.count()


def test_predict_local_result(df, spark):
    df_train, df_test = df[0], df[1]
    forecast_flow = ForecastFlowML(
        group_col="group",
        id_col="id",
        date_col="date",
        target_col="target",
        date_frequency="days",
        model_horizon=1,
        max_forecast_horizon=2,
        model=LGBMRegressor(),
    )
    forecast_flow.train(df_train, local_result=True)
    predictions = forecast_flow.predict(df_test, spark=spark)
    assert predictions.count() == df_test.count()


def test_predict_pandas_dataframe(df, spark):
    df_train_pandas, df_test_pandas = df[0].toPandas(), df[1].toPandas()
    forecast_flow = ForecastFlowML(
        group_col="group",
        id_col="id",
        date_col="date",
        target_col="target",
        date_frequency="days",
        model_horizon=1,
        max_forecast_horizon=2,
        model=LGBMRegressor(),
    )
    forecast_flow.train(df_train_pandas, spark=spark)
    predictions = forecast_flow.predict(df_test_pandas, spark=spark)
    assert len(predictions) == len(df_test_pandas)


def test_cross_validate(df):
    df_train = df[0]
    n_group = df_train.select("group").dropDuplicates().count()
    n_cv_splits = 1
    model_horizon = 1
    max_forecast_horizon = 2
    forecast_flow = ForecastFlowML(
        group_col="group",
        id_col="id",
        date_col="date",
        target_col="target",
        date_frequency="days",
        model_horizon=model_horizon,
        max_forecast_horizon=max_forecast_horizon,
        model=LGBMRegressor(),
    )
    cv_result = forecast_flow.cross_validate(df_train, n_cv_splits=n_cv_splits)
    assert cv_result.count() == n_cv_splits * max_forecast_horizon * n_group


def test_cross_validate_pandas_dataframe(df, spark):
    df_train_pandas = df[0].toPandas()
    n_group = len(df_train_pandas["id"].unique())
    n_cv_splits = 1
    model_horizon = 1
    max_forecast_horizon = 2
    forecast_flow = ForecastFlowML(
        group_col="group",
        id_col="id",
        date_col="date",
        target_col="target",
        date_frequency="days",
        model_horizon=model_horizon,
        max_forecast_horizon=max_forecast_horizon,
        model=LGBMRegressor(),
    )
    cv_result = forecast_flow.cross_validate(
        df_train_pandas, n_cv_splits=n_cv_splits, spark=spark
    )
    assert len(cv_result) == n_cv_splits * max_forecast_horizon * n_group


def test_grid_search(df):
    df_train = df[0]
    n_group = df_train.select("group").dropDuplicates().count()
    param_grid = {"n_estimators": [1, 2]}
    forecast_flow = ForecastFlowML(
        group_col="group",
        id_col="id",
        date_col="date",
        target_col="target",
        date_frequency="days",
        model_horizon=1,
        max_forecast_horizon=2,
        model=LGBMRegressor(),
    )
    results = forecast_flow.grid_search(df_train, param_grid=param_grid, n_cv_splits=1)
    assert len(results) == n_group * len(param_grid["n_estimators"])


def test_feature_importance(df):
    df_train = df[0]
    n_group = df_train.select("group").dropDuplicates().count()
    model_horizon = 1
    max_forecast_horizon = 2
    n_horizon = max_forecast_horizon / model_horizon
    forecast_flow = ForecastFlowML(
        group_col="group",
        id_col="id",
        date_col="date",
        target_col="target",
        date_frequency="days",
        model_horizon=model_horizon,
        max_forecast_horizon=max_forecast_horizon,
        model=LGBMRegressor(),
    )
    trained_models = forecast_flow.train(df_train)
    feature_importance = forecast_flow.get_feature_importance(trained_models)
    assert len(feature_importance) == n_group * n_horizon


def test_feature_importance_local_result(df):
    df_train = df[0]
    n_group = df_train.select("group").dropDuplicates().count()
    model_horizon = 1
    max_forecast_horizon = 2
    n_horizon = max_forecast_horizon / model_horizon
    forecast_flow = ForecastFlowML(
        group_col="group",
        id_col="id",
        date_col="date",
        target_col="target",
        date_frequency="days",
        model_horizon=model_horizon,
        max_forecast_horizon=max_forecast_horizon,
        model=LGBMRegressor(),
    )
    forecast_flow.train(df_train, local_result=True)
    feature_importance = forecast_flow.get_feature_importance()
    assert len(feature_importance) == n_group * n_horizon


def test_feature_importance_pandas_dataframe(df, spark):
    df_train_pandas = df[0].toPandas()
    n_group = len(df_train_pandas["group"].unique())
    model_horizon = 1
    max_forecast_horizon = 2
    n_horizon = max_forecast_horizon / model_horizon
    forecast_flow = ForecastFlowML(
        group_col="group",
        id_col="id",
        date_col="date",
        target_col="target",
        date_frequency="days",
        model_horizon=model_horizon,
        max_forecast_horizon=max_forecast_horizon,
        model=LGBMRegressor(),
    )
    forecast_flow.train(df_train_pandas, spark=spark)
    feature_importance = forecast_flow.get_feature_importance()
    assert len(feature_importance) == n_group * n_horizon
