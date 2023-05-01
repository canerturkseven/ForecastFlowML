import re
import copy
import pandas as pd


class _DirectForecaster:
    def __init__(
        self,
        id_col,
        group_col,
        date_col,
        target_col,
        model,
        model_horizon,
        max_forecast_horizon,
        categorical_cols=None,
        use_lag_range=0,
    ):
        self.id_col = id_col
        self.group_col = group_col
        self.date_col = date_col
        self.target_col = target_col
        self.categorical_cols = categorical_cols
        self.model = model
        self.model_horizon = model_horizon
        self.use_lag_range = use_lag_range
        self.n_horizon = max_forecast_horizon // model_horizon

    def _convert_categorical(self, df):
        categorical_cols = self.categorical_cols
        if categorical_cols is not None:
            df[categorical_cols] = df[categorical_cols].astype("category")
        return df

    def _filter_horizon(self, df, forecast_horizon):
        dates = df[self.date_col].sort_values().unique()
        forecast_dates = dates[[fh - 1 for fh in forecast_horizon]]
        return df[df[self.date_col].isin(forecast_dates)].copy()

    def _forecast_horizon(self, i):
        model_horizon = self.model_horizon
        return tuple(list(range(i * model_horizon + 1, (i + 1) * model_horizon + 1)))

    def _filter_features(self, df, forecast_horizon):
        min_lag = max(forecast_horizon)
        lag_range = self.use_lag_range
        feature_cols = [
            col
            for col in df.select_dtypes(["number", "category"]).columns
            if col not in [self.id_col, self.group_col, self.date_col, self.target_col]
        ]
        lag_cols = [
            col
            for col in feature_cols
            if re.findall("(^|_)lag_\d+(_|$)", col, re.IGNORECASE)
        ]
        keep_lags_str = "|".join(map(str, range(min_lag, min_lag + lag_range + 1)))
        keep_lags = [
            col
            for col in lag_cols
            if re.findall(
                f"^lag_({keep_lags_str})$|_lag_{min_lag}(_|$)", col, re.IGNORECASE
            )
        ]
        features = list(set(feature_cols) - set(lag_cols)) + keep_lags
        return features

    def fit(self, df):
        df = df.copy()
        df = self._convert_categorical(df)
        group = df[self.group_col].iloc[0]
        group_model = self.model[group] if isinstance(self.model, dict) else self.model

        self.model_ = {}
        for i in range(self.n_horizon):
            forecast_horizon = self._forecast_horizon(i)
            features = self._filter_features(df, forecast_horizon)

            X = df[features]
            y = df[self.target_col]

            horizon_model = copy.deepcopy(group_model)
            horizon_model.fit(X, y)
            self.model_[forecast_horizon] = horizon_model

        return self

    def predict(self, df):
        df = df.copy()
        df = self._convert_categorical(df)
        group = df[self.group_col].iloc[0]

        result_list = []
        for forecast_horizon, model in self.model_.items():
            features = self._filter_features(df, forecast_horizon)
            model_data = self._filter_horizon(df, forecast_horizon)

            model_data["prediction"] = model.predict(model_data[features])
            result_list.append(model_data)

        prediction = pd.concat(result_list).reset_index(drop=True)
        prediction["group"] = group
        prediction = prediction.rename(
            columns={self.id_col: "id", self.date_col: "date"}
        )
        prediction = prediction.loc[:, ["group", "id", "date", "prediction"]]

        return prediction
