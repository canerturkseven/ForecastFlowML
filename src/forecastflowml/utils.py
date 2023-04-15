import pandas as pd
import pyspark


def _check_input_type(df: pd.DataFrame):
    if isinstance(df, pd.DataFrame):
        return "df_pandas"
    elif isinstance(df, pyspark.sql.dataframe.DataFrame):
        return "df_spark"
    else:
        raise NotImplementedError(
            "Input is expected to be a pandas.DataFrame or"
            " pyspark.sql.dataframe.DataFrame"
        )


def _check_spark(model, input_type, spark):
    if (input_type == "df_pandas") & (spark is None):
        raise ValueError("spark instance must be supplied in case of Pandas DataFrame")


def _check_fitted(model, trained_models, spark):
    if (not hasattr(model, "model_")) & (trained_models is None):
        raise ValueError("train method should be called before predict")
    if (trained_models is None) & (hasattr(model, "model_")) & ((spark is None)):
        raise ValueError("spark instance must be supplied")
