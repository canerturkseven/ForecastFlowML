import pyspark.sql.functions as F
from pyspark.sql.window import Window
import pandas as pd


def history_length(df, id_col, date_col):
    w = Window.partitionBy(id_col).orderBy(date_col)
    df = df.withColumn("history_length", F.row_number().over(w))
    return df


def lag_window_summarizer(df, id_col, target_col, date_col, features):

    w1 = Window.partitionBy(id_col).orderBy(date_col)
    for key, values in features.items():
        if key == "lag":
            for lag in values:
                df = df.withColumn(f"lag_{lag}", F.lag(target_col, lag).over(w1))
        else:
            for window, lag in values:
                w2 = w1.rowsBetween(-(lag + window), -lag)
                df = df.withColumn(
                    f"window_{window}_lag_{lag}_{key}",
                    F.expr(f"{key}({target_col})").over(w2),
                )
    return df


def count_consecutive_values(df, id_col, value_col, date_col, value, lags):

    w1 = (
        Window.partitionBy(id_col)
        .orderBy(date_col)
        .rowsBetween(Window.unboundedPreceding, 0)
    )
    w2 = (
        Window.partitionBy(id_col, "group")
        .orderBy(date_col)
        .rowsBetween(Window.unboundedPreceding, 0)
    )
    df = (
        df.withColumn("mask", F.when(F.col(value_col) == value, 1).otherwise(0))
        .withColumn("group", F.sum(1 - F.col("mask")).over(w1))
        .withColumn("count", F.sum("mask").over(w2))
    )

    w3 = Window.partitionBy(id_col).orderBy(date_col)
    for lag in lags:
        output_col = f"count_consecutive_value_lag_{lag}"
        df = df.withColumn(output_col, F.lag("count", lag).over(w3))
    df = df.drop("mask", "group", "count")
    return df


def date_features(df, date_col, features):
    supported_features = [
        "day_of_week",
        "day_of_year",
        "day_of_month",
        "week_of_year",
        "week_of_month",
        "weekend",
        "month",
        "quarter",
        "year",
    ]

    not_supported = set(features) - set(supported_features)
    if len(not_supported) > 0:
        raise ValueError(f"{', '.join(not_supported)} feature(s) not supported.")

    for feature in features:
        if feature == "day_of_week":
            df = df.withColumn(feature, F.dayofweek(F.col(date_col)).cast("tinyint"))
        if feature == "day_of_year":
            df = df.withColumn(feature, F.dayofyear(F.col(date_col)).cast("smallint"))
        if feature == "day_of_month":
            df = df.withColumn(feature, F.dayofmonth(F.col(date_col)).cast("tinyint"))
        if feature == "week_of_year":
            df = df.withColumn(feature, F.weekofyear(F.col(date_col)).cast("tinyint"))
        if feature == "week_of_month":
            df = df.withColumn(
                feature, F.ceil(F.dayofmonth(F.col(date_col)) / 7).cast("tinyint")
            )
        if feature == "weekend":
            df = df.withColumn(
                feature,
                F.when(F.dayofweek(F.col(date_col)).isin([1, 7]), 1)
                .otherwise(0)
                .cast("tinyint"),
            )
        if feature == "month":
            df = df.withColumn("month", F.month(F.col(date_col)).cast("tinyint"))
        if feature == "quarter":
            df = df.withColumn("quarter", F.quarter(F.col(date_col)).cast("tinyint"))
        if feature == "year":
            df = df.withColumn("year", F.year(F.col(date_col)).cast("smallint"))
    return df


class FeatureExtractor:
    def __init__(
        self,
        id_col,
        date_col,
        target_col,
        lag_window_features=None,
        date_features=None,
        encode_events=None,
        count_consecutive_values=None,
        history_length=False,
        spark=None,
    ):
        self.id_col = id_col
        self.date_col = date_col
        self.target_col = target_col
        self.lag_window_features = lag_window_features
        self.date_features = date_features
        self.encode_events = encode_events
        self.count_consecutive_values = count_consecutive_values
        self.history_length = history_length
        self.spark = spark

    def transform(self, df):
        if isinstance(df, pd.DataFrame):
            type_pandas = True
            df = self.spark.createDataFrame(df)
        else:
            type_pandas = False
        if self.lag_window_features is not None:
            df = lag_window_summarizer(
                df,
                self.id_col,
                self.target_col,
                self.date_col,
                self.lag_window_features,
            )
        if self.count_consecutive_values is not None:
            df = count_consecutive_values(
                df,
                self.id_col,
                self.target_col,
                self.date_col,
                self.count_consecutive_values["value"],
                self.count_consecutive_values["lags"],
            )
        if self.history_length:
            df = history_length(df, self.id_col, self.date_col)
        if self.date_features is not None:
            df = date_features(df, self.date_col, self.date_features)

        df = df.toPandas() if type_pandas else df
        return df
