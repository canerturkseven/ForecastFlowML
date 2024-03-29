import sklearn
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


def _score_func(y_true, y_pred, metric):
    sklearn_scorer = sklearn.metrics.get_scorer(metric)
    return sklearn_scorer._sign * sklearn_scorer._score_func(
        y_true=y_true, y_pred=y_pred, **sklearn_scorer._kwargs
    )


def _cross_val_predict(
    *,
    forecaster,
    df,
    cv,
    refit,
):
    group_col = forecaster.group_col
    id_col = forecaster.id_col
    date_col = forecaster.date_col
    target_col = forecaster.target_col

    cv_predictions_list = []
    for i, fold in enumerate(cv):
        train_idx, test_idx = fold[0], fold[1]
        df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]

        if refit or i == 0:
            forecaster = forecaster.fit(df_train)

        prediction = forecaster.predict(df_test)
        prediction["cv"] = str(i)
        prediction = prediction.merge(
            df_test[[id_col, date_col, target_col]],
            on=[id_col, date_col],
            how="left",
        )
        cv_predictions_list.append(prediction)

    cv_predictions = pd.concat(cv_predictions_list).reset_index(drop=True)
    cv_predictions = cv_predictions[
        [group_col, id_col, date_col, "cv", target_col, "prediction"]
    ]

    return cv_predictions


class _TimeBasedSplit:
    def __init__(
        self,
        *,
        date_frequency,
        date_col,
        forecast_horizon=None,
        max_train_size=None,
        n_splits=5,
        end_offset=0,
        step_length=None,
    ):
        self.forecast_horizon = forecast_horizon
        self.max_train_size = max_train_size
        self.n_splits = n_splits
        self.date_frequency = date_frequency
        self.end_offset = end_offset
        self.date_col = date_col
        self.step_length = step_length

    @property
    def forecast_horizon(self):
        return self._forecast_horizon

    @forecast_horizon.setter
    def forecast_horizon(self, value):
        if value is None:
            self._forecast_horizon = [1]
        elif any(i < 0 for i in value):
            raise ValueError("all forecast_horizon values must be positive")
        else:
            self._forecast_horizon = value

    @property
    def max_train_size(self):
        return self._max_train_size

    @max_train_size.setter
    def max_train_size(self, value):
        if value:
            if int(value) <= 0:
                raise ValueError(
                    f"max_train_size must be positive, received {value} instead"
                )
        self._max_train_size = value

    @property
    def n_splits(self):
        return self._n_splits

    @n_splits.setter
    def n_splits(self, value):
        if int(value) <= 0:
            raise ValueError(f"n_splits must be positive, received {value} instead")
        else:
            self._n_splits = value

    @property
    def date_frequency(self):
        return self._date_frequency

    @date_frequency.setter
    def date_frequency(self, value):
        supported_date_frequency = [
            "years",
            "months",
            "weeks",
            "days",
            "hours",
            "minutes",
            "seconds",
            "microseconds",
        ]
        if value not in supported_date_frequency:
            raise ValueError(
                f"{value} is not supported as date frequency, "
                "supported frequencies are: {' '.join(supported_date_frequency)}"
            )
        else:
            self._date_frequency = value

    @property
    def end_offset(self):
        return self._end_offset

    @end_offset.setter
    def end_offset(self, value):
        if int(value) < 0:
            raise ValueError(f"end_offset must be >= 0, received {value} instead")
        else:
            self._end_offset = value

    @property
    def step_length(self):
        return self._step_length

    @step_length.setter
    def step_length(self, value):
        if value is None:
            self._step_length = max(self.forecast_horizon)
        elif int(value) < 0:
            raise ValueError(f"step_length must be > 0, received {value} instead")
        else:
            self._step_length = value

    def _check_input(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")

    def _check_date_col(self, df):
        if self.date_col not in df.columns:
            raise ValueError(f"{self.date_col} is not in the {df} columns")
        if not np.issubdtype(df[self.date_col].dtypes, np.datetime64):
            raise ValueError(f"{self.date_col} must be a date column")

    def _check_n_splits(self, df):
        max_date = df[self.date_col].max().to_pydatetime()
        max_iter = int(
            self.end_offset
            + max(self.forecast_horizon)
            + ((self.n_splits - 1) * self.step_length)
        )
        train_end_date = max_date - relativedelta(**{self.date_frequency: max_iter})

        if df[df[self.date_col] <= train_end_date].empty:
            raise ValueError(
                f"Too many splits={self.n_splits} "
                f"with forecast horizon={self.forecast_horizon}, and "
                f"step_length={self.step_length} "
                "for the date sequence."
            )

    def split(self, df):
        max_train_size = self.max_train_size
        date_frequency = self.date_frequency
        forecast_horizon = self.forecast_horizon
        n_splits = self.n_splits
        end_offset = self.end_offset
        date_col = self.date_col
        step_length = self.step_length

        self._check_input(df)
        self._check_date_col(df)
        self._check_n_splits(df)

        df = df.reset_index(drop=True)
        max_date = df[date_col].max().to_pydatetime()
        splits = []
        for i in range(n_splits):
            train_end = max_date - relativedelta(
                **{
                    date_frequency: (
                        i * step_length + end_offset + int(max(forecast_horizon))
                    )
                }
            )
            test_dates = [
                train_end + fh * relativedelta(**{date_frequency: 1})
                for fh in forecast_horizon
            ]
            test_condition = df[date_col].isin(test_dates)

            if self.max_train_size:
                train_start = train_end - relativedelta(
                    **{date_frequency: max_train_size}
                )
                train_condition = (df[date_col] <= train_end) & (
                    df[date_col] > train_start
                )
            else:
                train_condition = df[date_col] <= train_end

            splits.append(
                (
                    df[train_condition].index.tolist(),
                    df[test_condition].index.tolist(),
                )
            )
        return splits
