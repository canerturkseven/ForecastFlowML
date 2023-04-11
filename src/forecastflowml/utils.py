import pandas as pd
import pyspark


def _check_input_type(df):
    if isinstance(df, pd.DataFrame):
        return "df_pandas"
    elif isinstance(df, pyspark.sql.dataframe.DataFrame):
        return "df_spark"
    else:
        raise NotImplementedError(
            "Input is expected to be a pandas.DataFrame or pyspark.sql.dataframe.DataFrame"
        )


def _check_spark(model, input_type, spark):
    if ((input_type == "df_pandas") | hasattr(model, "model_")) & (spark == None):
        raise ValueError("spark instance must be supplied in case of Pandas DataFrame.")
