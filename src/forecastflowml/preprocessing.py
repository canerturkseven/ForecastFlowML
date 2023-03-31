import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.ml import Transformer, Estimator, Model, Pipeline
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark import keyword_only


class FitTransformPipeline(Pipeline):
    def fit_transform(self, dataset):
        stages = self.getStages()
        for stage in stages:
            if not (isinstance(stage, Estimator) or isinstance(stage, Transformer)):
                raise TypeError(
                    "Cannot recognize a pipeline stage of type %s." % type(stage)
                )
        for stage in stages:
            if isinstance(stage, Transformer):
                dataset = stage.transform(dataset)
            else:  # must be an Estimator
                model = stage.fit(dataset)
                dataset = model.transform(dataset)
        return dataset


class HistoryLength(Estimator):
    idCol = Param(
        Params._dummy(),
        "idCol",
        "id column",
        typeConverter=TypeConverters.toString,
    )
    dateCol = Param(
        Params._dummy(),
        "dateCol",
        "date column",
        typeConverter=TypeConverters.toString,
    )

    @keyword_only
    def __init__(
        self,
        idCol=None,
        dateCol=None,
    ):
        super().__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(
        self,
        idCol=None,
        dateCol=None,
    ):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def getIdCol(self):
        return self.getOrDefault(self.idCol)

    def getDateCol(self):
        return self.getOrDefault(self.dateCol)

    def setIdCol(self, id_col):
        return self.setParams(idCol=id_col)

    def setDateCol(self, date_col):
        return self.setParams(dateCol=date_col)

    def _fit(self, df):
        return HistoryLengthModel(
            idCol=self.getIdCol(),
            dateCol=self.getDateCol(),
            df_=df,
        )


class HistoryLengthModel(Model):
    idCol = Param(
        Params._dummy(),
        "idCol",
        "id column",
        typeConverter=TypeConverters.toString,
    )
    dateCol = Param(
        Params._dummy(),
        "dateCol",
        "date column",
        typeConverter=TypeConverters.toString,
    )
    df_ = Param(
        Params._dummy(),
        "df_",
        "dataset from fitting phase",
    )

    @keyword_only
    def __init__(
        self,
        idCol=None,
        dateCol=None,
        df_=None,
    ):
        super().__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(
        self,
        idCol=None,
        dateCol=None,
        df_=None,
    ):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def getIdCol(self):
        return self.getOrDefault(self.idCol)

    def getDateCol(self):
        return self.getOrDefault(self.dateCol)

    def getDf_(self):
        return self.getOrDefault(self.df_)

    def setDateCol(self, date_col):
        return self.setParams(dateCol=date_col)

    def setIdCol(self, id_col):
        return self.setParams(idCol=id_col)

    def setDf_(self, df_):
        return self.setParams(df_=df_)

    def _transform(self, df):
        id_col = self.getIdCol()
        date_col = self.getDateCol()
        df_ = self.getDf_()
        key_cols = [date_col, id_col]
        non_key_cols = set(df.columns) - set(key_cols)
        df = df_.alias("df_fit").join(
            df.alias("df_transform"), on=key_cols, how="outer"
        )
        for col in non_key_cols:
            df = (
                df.withColumn(
                    f"{col}_",
                    F.coalesce(f"df_fit.{col}", f"df_transform.{col}"),
                )
                .drop(col)
                .withColumnRenamed(f"{col}_", col)
            )

        w = Window.partitionBy(id_col).orderBy(date_col)
        df = df.withColumn("history_length", F.row_number().over(w))

        return df


class LagWindowSummarizer(Estimator):
    idCol = Param(
        Params._dummy(),
        "idCol",
        "id column",
        typeConverter=TypeConverters.toString,
    )
    targetCol = Param(
        Params._dummy(),
        "targetCol",
        "target column",
        typeConverter=TypeConverters.toString,
    )
    dateCol = Param(
        Params._dummy(),
        "dateCol",
        "date column",
        typeConverter=TypeConverters.toString,
    )
    features = Param(
        Params._dummy(),
        "features",
        "features",
    )

    @keyword_only
    def __init__(
        self,
        idCol=None,
        targetCol=None,
        dateCol=None,
        features=None,
    ):
        super().__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(
        self,
        idCol=None,
        targetCol=None,
        dateCol=None,
        features=None,
    ):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def getIdCol(self):
        return self.getOrDefault(self.idCol)

    def getTargetCol(self):
        return self.getOrDefault(self.targetCol)

    def getDateCol(self):
        return self.getOrDefault(self.dateCol)

    def getFeatures(self):
        return self.getOrDefault(self.features)

    def setTargetCol(self, target_col):
        return self.setParams(targetCol=target_col)

    def setDateCol(self, date_col):
        return self.setParams(dateCol=date_col)

    def setFeatures(self, features):
        return self.setParams(features=features)

    def setIdCol(self, id_col):
        return self.setParams(idCol=id_col)

    def _fit(self, df):
        return LagWindowSummarizerModel(
            idCol=self.getIdCol(),
            targetCol=self.getTargetCol(),
            dateCol=self.getDateCol(),
            features=self.getFeatures(),
            df_=df,
        )


class LagWindowSummarizerModel(Model):
    idCol = Param(
        Params._dummy(),
        "idCol",
        "id column",
        typeConverter=TypeConverters.toString,
    )
    targetCol = Param(
        Params._dummy(),
        "targetCol",
        "target column",
        typeConverter=TypeConverters.toString,
    )
    dateCol = Param(
        Params._dummy(),
        "dateCol",
        "date column",
        typeConverter=TypeConverters.toString,
    )
    features = Param(
        Params._dummy(),
        "features",
        "features",
    )
    df_ = Param(
        Params._dummy(),
        "df_",
        "dataset from fitting phase",
    )

    @keyword_only
    def __init__(
        self,
        idCol=None,
        targetCol=None,
        dateCol=None,
        features=None,
        df_=None,
    ):
        super().__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(
        self,
        idCol=None,
        targetCol=None,
        dateCol=None,
        features=None,
        df_=None,
    ):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def getIdCol(self):
        return self.getOrDefault(self.idCol)

    def getDf_(self):
        return self.getOrDefault(self.df_)

    def getTargetCol(self):
        return self.getOrDefault(self.targetCol)

    def getDateCol(self):
        return self.getOrDefault(self.dateCol)

    def getFeatures(self):
        return self.getOrDefault(self.features)

    def setIdCol(self, id_col):
        return self.setParams(idCol=id_col)

    def setDf_(self, df_):
        return self.setParams(df_=df_)

    def setTargetCol(self, target_col):
        return self.setParams(targetCol=target_col)

    def setDateCol(self, date_col):
        return self.setParams(dateCol=date_col)

    def setFeatures(self, features):
        return self.setParams(features=features)

    def _transform(self, df):
        id_col = self.getIdCol()
        features = self.getFeatures()
        target_col = self.getTargetCol()
        date_col = self.getDateCol()

        key_cols = [self.getDateCol(), self.getIdCol()]
        non_key_cols = set(df.columns) - set(key_cols)
        df_ = self.getDf_()
        df = df_.alias("df_fit").join(
            df.alias("df_transform"), on=key_cols, how="outer"
        )
        for col in non_key_cols:
            df = (
                df.withColumn(
                    f"{col}_",
                    F.coalesce(f"df_fit.{col}", f"df_transform.{col}"),
                )
                .drop(col)
                .withColumnRenamed(f"{col}_", col)
            )

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


class TriangleEventEncoder(Estimator):
    idCol = Param(
        Params._dummy(),
        "idCol",
        "id column",
        typeConverter=TypeConverters.toString,
    )
    eventCols = Param(
        Params._dummy(),
        "eventCols",
        "event columns",
        typeConverter=TypeConverters.toListString,
    )
    dateCol = Param(
        Params._dummy(),
        "dateCol",
        "date column",
        typeConverter=TypeConverters.toString,
    )
    window = Param(
        Params._dummy(),
        "window",
        "window",
        typeConverter=TypeConverters.toInt,
    )

    @keyword_only
    def __init__(
        self,
        idCol=None,
        eventCols=None,
        dateCol=None,
        window=None,
    ):
        super().__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(
        self,
        idCol=None,
        eventCols=None,
        dateCol=None,
        window=None,
    ):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def getIdCol(self):
        return self.getOrDefault(self.idCol)

    def getWindow(self):
        return self.getOrDefault(self.window)

    def getEventCols(self):
        return self.getOrDefault(self.eventCols)

    def getDateCol(self):
        return self.getOrDefault(self.dateCol)

    def setEventCols(self, event_cols):
        return self.setParams(eventCols=event_cols)

    def setWindow(self, window):
        return self.setParams(window=window)

    def setDateCol(self, date_col):
        return self.setParams(dateCol=date_col)

    def setIdCol(self, id_col):
        return self.setParams(idCol=id_col)

    def _fit(self, df):
        return TriangleEventEncoderModel(
            idCol=self.getIdCol(),
            eventCols=self.getEventCols(),
            dateCol=self.getDateCol(),
            window=self.getWindow(),
            df_=df,
        )


class TriangleEventEncoderModel(Model):
    idCol = Param(
        Params._dummy(),
        "idCol",
        "id column",
        typeConverter=TypeConverters.toString,
    )
    eventCols = Param(
        Params._dummy(),
        "eventCols",
        "event columns",
        typeConverter=TypeConverters.toListString,
    )
    dateCol = Param(
        Params._dummy(),
        "dateCol",
        "date column",
        typeConverter=TypeConverters.toString,
    )
    window = Param(
        Params._dummy(),
        "window",
        "window",
        typeConverter=TypeConverters.toInt,
    )
    df_ = Param(
        Params._dummy(),
        "df_",
        "dataset from fitting phase",
    )

    @keyword_only
    def __init__(
        self,
        idCol=None,
        eventCols=None,
        dateCol=None,
        window=None,
        df_=None,
    ):
        super().__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(
        self,
        idCol=None,
        eventCols=None,
        dateCol=None,
        window=None,
        df_=None,
    ):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def getIdCol(self):
        return self.getOrDefault(self.idCol)

    def getDf_(self):
        return self.getOrDefault(self.df_)

    def getEventCols(self):
        return self.getOrDefault(self.eventCols)

    def getDateCol(self):
        return self.getOrDefault(self.dateCol)

    def getWindow(self):
        return self.getOrDefault(self.window)

    def setIdCol(self, id_col):
        return self.setParams(idCol=id_col)

    def setDf_(self, df_):
        return self.setParams(df_=df_)

    def setEventCols(self, event_cols):
        return self.setParams(eventCols=event_cols)

    def setDateCol(self, date_col):
        return self.setParams(dateCol=date_col)

    def setWindow(self, window):
        return self.setParams(window=window)

    def _period_diff(self, first_date, second_date):
        data_frequency = self.getDateFrequency()
        if data_frequency == "day":
            return F.datediff(first_date, second_date)
        if data_frequency == "week":
            return F.datediff(first_date, second_date) / 7
        if data_frequency == "month":
            return F.months_between(first_date, second_date)

    def _transform(self, df):
        id_col = self.getIdCol()
        date_col = self.getDateCol()
        event_cols = self.getEventCols()
        window = self.getWindow()
        df_ = self.getDf_()

        key_cols = [id_col, date_col]
        non_key_cols = set(df.columns) - set(key_cols)
        df = df_.alias("df_fit").join(
            df.alias("df_transform"), on=key_cols, how="outer"
        )
        for col in non_key_cols:
            df = (
                df.withColumn(
                    f"{col}_",
                    F.coalesce(f"df_fit.{col}", f"df_transform.{col}"),
                )
                .drop(col)
                .withColumnRenamed(f"{col}_", col)
            )

        w = Window.partitionBy(id_col).orderBy(date_col)
        previous_rows = w.rowsBetween(Window.unboundedPreceding, 0)
        following_rows = w.rowsBetween(0, Window.unboundedFollowing)

        for event_col in event_cols:
            df = (
                df.withColumn(
                    "last_event_date",
                    F.last(
                        F.when(F.col(event_col) == 1, F.row_number().over(w)),
                        ignorenulls=True,
                    ).over(previous_rows),
                )
                .withColumn(
                    "next_event_date",
                    F.first(
                        F.when(F.col(event_col) == 1, F.row_number().over(w)),
                        ignorenulls=True,
                    ).over(following_rows),
                )
                .withColumn(
                    "periods_to_event",
                    F.col("next_event_date") - F.row_number().over(w),
                )
                .withColumn(
                    "periods_after_event",
                    F.col("last_event_date") - F.row_number().over(w),
                )
                .withColumn(
                    "periods_to_event",
                    F.when(F.col("periods_to_event") > window, None).otherwise(
                        F.col("periods_to_event")
                    ),
                )
                .withColumn(
                    "periods_after_event",
                    F.when(F.col("periods_after_event") < -window, None).otherwise(
                        F.col("periods_after_event")
                    ),
                )
                .withColumn(
                    event_col, F.greatest("periods_to_event", "periods_after_event")
                )
                .drop(
                    "last_event_date",
                    "next_event_date",
                    "periods_to_event",
                    "periods_after_event",
                )
            )
        return df


class CountConsecutiveValues(Estimator):
    idCol = Param(
        Params._dummy(),
        "idCol",
        "id column",
        typeConverter=TypeConverters.toString,
    )
    valueCol = Param(
        Params._dummy(),
        "valueCol",
        "value column",
        typeConverter=TypeConverters.toString,
    )
    dateCol = Param(
        Params._dummy(),
        "dateCol",
        "date column",
        typeConverter=TypeConverters.toString,
    )
    value = Param(
        Params._dummy(),
        "value",
        "value to count consecutively",
        typeConverter=TypeConverters.toFloat,
    )
    lags = Param(
        Params._dummy(),
        "lags",
        "lags",
        typeConverter=TypeConverters.toListInt,
    )

    @keyword_only
    def __init__(
        self,
        idCol=None,
        valueCol=None,
        dateCol=None,
        value=None,
        lags=None,
    ):
        super().__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(
        self,
        idCol=None,
        valueCol=None,
        dateCol=None,
        value=None,
        lags=None,
    ):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def getIdCol(self):
        return self.getOrDefault(self.idCol)

    def getValueCol(self):
        return self.getOrDefault(self.valueCol)

    def getDateCol(self):
        return self.getOrDefault(self.dateCol)

    def getValue(self):
        return self.getOrDefault(self.value)

    def getLags(self):
        return self.getOrDefault(self.lags)

    def setIdCol(self, id_col):
        return self.setParams(idCol=id_col)

    def setValueCol(self, value_col):
        return self.setParams(valueCol=value_col)

    def setDateCol(self, date_col):
        return self.setParams(dateCol=date_col)

    def setValue(self, value):
        return self.setParams(value=value)

    def setLags(self, lags):
        return self.setParams(lags=lags)

    def _fit(self, df):
        return CountConsecutiveValuesModel(
            idCol=self.getIdCol(),
            valueCol=self.getValueCol(),
            dateCol=self.getDateCol(),
            value=self.getValue(),
            lags=self.getLags(),
            df_=df,
        )


class CountConsecutiveValuesModel(Model):
    idCol = Param(
        Params._dummy(),
        "idCol",
        "id column",
        typeConverter=TypeConverters.toString,
    )
    valueCol = Param(
        Params._dummy(),
        "valueCol",
        "value column",
        typeConverter=TypeConverters.toString,
    )
    dateCol = Param(
        Params._dummy(),
        "dateCol",
        "date column",
        typeConverter=TypeConverters.toString,
    )
    value = Param(
        Params._dummy(),
        "value",
        "value to count consecutively",
        typeConverter=TypeConverters.toFloat,
    )
    lags = Param(
        Params._dummy(),
        "lags",
        "lags",
        typeConverter=TypeConverters.toListInt,
    )
    df_ = Param(
        Params._dummy(),
        "df_",
        "dataset from fitting phase",
    )

    @keyword_only
    def __init__(
        self,
        idCol=None,
        valueCol=None,
        dateCol=None,
        value=None,
        lags=None,
        df_=None,
    ):
        super().__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(
        self,
        idCol=None,
        valueCol=None,
        dateCol=None,
        value=None,
        lags=None,
        df_=None,
    ):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def getDf_(self):
        return self.getOrDefault(self.df_)

    def setDf_(self, df_):
        return self.setParams(df_=df_)

    def getIdCol(self):
        return self.getOrDefault(self.idCol)

    def getValueCol(self):
        return self.getOrDefault(self.valueCol)

    def getDateCol(self):
        return self.getOrDefault(self.dateCol)

    def getValue(self):
        return self.getOrDefault(self.value)

    def getLags(self):
        return self.getOrDefault(self.lags)

    def setIdCol(self, id_col):
        return self.setParams(idCol=id_col)

    def setValueCol(self, value_col):
        return self.setParams(valueCol=value_col)

    def setDateCol(self, date_col):
        return self.setParams(dateCol=date_col)

    def setValue(self, value):
        return self.setParams(value=value)

    def setLags(self, lags):
        return self.setParams(lags=lags)

    def _transform(self, df):
        id_col = self.getIdCol()
        value_col = self.getValueCol()
        date_col = self.getDateCol()
        value = self.getValue()
        lags = self.getLags()
        df_ = self.getDf_()

        key_cols = [date_col, id_col]
        non_key_cols = set(df.columns) - set(key_cols)
        df = df_.alias("df_fit").join(
            df.alias("df_transform"), on=key_cols, how="outer"
        )
        for col in non_key_cols:
            df = (
                df.withColumn(
                    f"{col}_",
                    F.coalesce(f"df_fit.{col}", f"df_transform.{col}"),
                )
                .drop(col)
                .withColumnRenamed(f"{col}_", col)
            )

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


class DateFeatures(Transformer):
    dateCol = Param(
        Params._dummy(),
        "dateCol",
        "date column",
        typeConverter=TypeConverters.toString,
    )
    features = Param(
        Params._dummy(),
        "features",
        "date features to create",
        typeConverter=TypeConverters.toListString,
    )
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

    @keyword_only
    def __init__(self, dateCol=None, features=None):
        super().__init__()
        self._setDefault(features=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, dateCol=None, features=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setFeatures(self, features):
        return self.setParams(features=features)

    def getFeatures(self):
        return self.getOrDefault(self.features)

    def setDateCol(self, date_col):
        return self.setParams(dateCol=date_col)

    def getDateCol(self):
        return self.getOrDefault(self.dateCol)

    def _check_input(self):
        if not self.isSet("dateCol"):
            raise ValueError("Date column is not set.")

        not_supported = set(self.getFeatures()) - set(self.supported_features)
        if len(not_supported) > 0:
            raise ValueError(f"{', '.join(not_supported)} feature(s) not supported.")

    def _transform(self, df):
        self._check_input()
        features = self.getFeatures()
        date_col = self.getDateCol()
        for feature in features:
            if feature == "day_of_week":
                df = df.withColumn(
                    feature, F.dayofweek(F.col(date_col)).cast("tinyint")
                )
            if feature == "day_of_year":
                df = df.withColumn(
                    feature, F.dayofyear(F.col(date_col)).cast("smallint")
                )
            if feature == "day_of_month":
                df = df.withColumn(
                    feature, F.dayofmonth(F.col(date_col)).cast("tinyint")
                )
            if feature == "week_of_year":
                df = df.withColumn(
                    feature, F.weekofyear(F.col(date_col)).cast("tinyint")
                )
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
                df = df.withColumn(
                    "quarter", F.quarter(F.col(date_col)).cast("tinyint")
                )
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
    ):
        self.id_col = id_col
        self.date_col = date_col
        self.target_col = target_col
        self.lag_window_features = lag_window_features
        self.date_features = date_features
        self.encode_events = encode_events
        self.count_consecutive_values = count_consecutive_values
        self.history_length = history_length

    def transform(self, df):
        stages = []
        if self.lag_window_features is not None:
            stages.append(
                LagWindowSummarizer(
                    idCol=self.id_col,
                    targetCol=self.target_col,
                    dateCol=self.date_col,
                    features=self.lag_window_features,
                )
            )
        if self.count_consecutive_values is not None:
            stages.append(
                CountConsecutiveValues(
                    idCol=self.id_col,
                    valueCol=self.target_col,
                    dateCol=self.date_col,
                    value=self.count_consecutive_values["value"],
                    lags=self.count_consecutive_values["lags"],
                )
            )
        if self.history_length:
            stages.append(
                HistoryLength(
                    idCol=self.id_col,
                    dateCol=self.date_col,
                )
            )
        if self.encode_events is not None:
            stages.append(
                TriangleEventEncoder(
                    idCol=self.id_col,
                    dateCol=self.date_col,
                    eventCols=self.encode_events["cols"],
                    window=self.encode_events["window"],
                )
            )
        if self.date_features is not None:
            stages.append(
                DateFeatures(
                    dateCol=self.date_col,
                    features=self.date_features,
                )
            )
        pipeline = FitTransformPipeline(stages=stages)
        return pipeline.fit_transform(df)
