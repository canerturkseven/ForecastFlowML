## ForecastFlowML: Scalable Machine Learning Forecasting with PySpark

[![Python Versions](https://img.shields.io/badge/python-3.7%20|%203.8%20|%203.9%20|%203.10%20|%203.11%20-blue)](https://www.python.org/downloads/) ![Tests](https://github.com/canerturkseven/ForecastFlowML/actions/workflows/tests.yml/badge.svg) [![codecov](https://codecov.io/github/canerturkseven/ForecastFlowML/branch/master/graph/badge.svg?token=DKAE8VSQ1M)](https://codecov.io/github/canerturkseven/ForecastFlowML) [![Documentation Status](https://readthedocs.org/projects/forecastflowml/badge/?version=latest)](https://forecastflowml.readthedocs.io/en/latest/?badge=latest)

ForecastFlowML is a scalable machine learning forecasting framework that enables parallel training (by distributing models rather than data) of scikit-learn like models based on PySpark.

With ForecastFlowMl, you can build scikit-learn like regressors as direct multi-step forecasters, and train a seperate model for each group in your dataset.
Our package leverages the power of PySpark to efficiently handle large datasets and enables distributed computing for faster model training.

## Features

ForecastFlowML provides a range of features that make it a powerful and flexible tool for time-series forecasting, including:

- Works with Pandas and Pyspark DataFrames.
- Distributed model training per group in the PySpark/Pandas DataFrames.
- Direct multi-step forecasting.
- Built-in time based cross-validation.
- Extensive time-series feature engineering (lag, rolling mean/std, stockout, history length).
- Hyperparameter tuning for each group model with grid search.
- Supports `scikit-learn` like libraries such as `LightGBM` or `XGBoost`.

Whether you're new to time-series forecasting or an experienced data scientist, ForecastFlowML can help you build and deploy accurate forecasting models at scale.

## Documentation

Reach to our latest documentation [here](https://forecastflowml.readthedocs.io/en/latest/).

### Get Started

[What is ForecastFlowML?](https://forecastflowml.readthedocs.io/en/latest/forecastflowml.html)

[Quick Start](https://forecastflowml.readthedocs.io/en/latest/notebooks/quick_start.html)

### User Guide

[Feature Engineering](https://forecastflowml.readthedocs.io/en/latest/notebooks/feature_engineering.html)

[Time Series Cross Validation](https://forecastflowml.readthedocs.io/en/latest/notebooks/cross_validation.html)

[Grid Search](https://forecastflowml.readthedocs.io/en/latest/notebooks/grid_search.html)

[Feature Importance](https://forecastflowml.readthedocs.io/en/latest/notebooks/feature_importance.html)

[Save/Load ForecastFlowML](https://forecastflowml.readthedocs.io/en/latest/notebooks/save_load.html)

## Installation

You can install the packaging using the following command.

```
pip install forecastflowml
```
