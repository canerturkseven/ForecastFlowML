import pandas as pd
import sklearn

def score_func(y_true, y_pred, metric):
    sklearn_scorer = sklearn.metrics.get_scorer(metric)
    return sklearn_scorer._sign * sklearn_scorer._score_func(
        y_true=y_true, y_pred=y_pred, **sklearn_scorer._kwargs
    )


def cross_val_forecast(model, df, id_col, feature_cols, date_col, target_col, cv):
    forecast = []
    for i, fold in enumerate(cv):

        train_idx, test_idx = fold[0], fold[1]
        df_train, df_test = df.iloc[train_idx], df.copy().iloc[test_idx]

        model.fit(df_train[feature_cols], df_train[target_col])
        df_test["forecast"] = model.predict(df_test[feature_cols])
        df_test["cv"] = str(i)

        forecast.append(df_test[[id_col, date_col, "cv", target_col, "forecast"]])

    cv_forecast = (
        pd.concat(forecast)
        .reset_index(drop=True)
        .rename(columns={id_col: "id", date_col: "date", target_col: "target"})
    )

    return cv_forecast
