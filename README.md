## ForecastFlowML: Scalable Machine Learning Forecasting with PySpark

[![Python Versions](https://img.shields.io/badge/python-3.7%20|%203.8%20|%203.9%20|%203.10%20|%203.11%20-blue)](https://www.python.org/downloads/) ![Tests](https://github.com/canerturkseven/ForecastFlowML/actions/workflows/tests.yml/badge.svg) [![codecov](https://codecov.io/github/canerturkseven/ForecastFlowML/branch/master/graph/badge.svg?token=DKAE8VSQ1M)](https://codecov.io/github/canerturkseven/ForecastFlowML) [![Documentation Status](https://readthedocs.org/projects/forecastflowml/badge/?version=latest)](https://forecastflowml.readthedocs.io/en/latest/?badge=latest)

ForecastFlowML is a scalable machine learning forecasting framework that enables parallel training (by distributing models rather than data) of scikit-learn like models based on PySpark.

With ForecastFlowML, you can build scikit-learn like regressors as direct multi-step forecasters, and train a seperate model for each group in your dataset.
Our package leverages the power of PySpark to efficiently handle large datasets and enables distributed computing for faster model training.

## Features

ForecastFlowML provides a range of features that make it a powerful and flexible tool for time-series forecasting, including:

- Scaleable and extensive time series feature engineering (lag, rolling mean/std, stockout, history length) with PySpark.
- Parallel model training per group in the dataset with Pyspark Pandas UDFs.
- Direct multi-step forecasting.
- Built-in time based cross-validation.
- Hyperparameter tuning for each group model with grid search.
- Supports `scikit-learn` like libraries such as `LightGBM` or `XGBoost`.

## Documentation

Reach out to our latest documentation [here](https://forecastflowml.readthedocs.io/en/latest/).

### User Guides

[What is ForecastFlowML?](https://forecastflowml.readthedocs.io/en/latest/forecastflowml.html)

[Feature Engineering](https://forecastflowml.readthedocs.io/en/latest/notebooks/feature_engineering.html)

[Time Series Cross Validation](https://forecastflowml.readthedocs.io/en/latest/notebooks/cross_validation.html)

[Grid Search](https://forecastflowml.readthedocs.io/en/latest/notebooks/grid_search.html)

[Feature Importance](https://forecastflowml.readthedocs.io/en/latest/notebooks/feature_importance.html)

[Save/Load ForecastFlowML](https://forecastflowml.readthedocs.io/en/latest/notebooks/save_load.html)

### Examples

[Kaggle Walmart M5 Forecasting Competition (18th solution)](https://www.kaggle.com/code/canerturkseven/forecastflowml-m5-forecasting-accuracy)

[Retail Demand Forecasting](https://forecastflowml.readthedocs.io/en/latest/notebooks/retail_demand_forecasting.html)

## Installation

### ForecastFlowML installation

You can install the package using the following command:

```
pip install forecastflowml
```

#### Check Java

Make sure you have installed Java 11. You can check whether you have Java or not with the following command:

```
java -version
```

#### Set PYSPARK_PYTHON

In the python script, set PYSPARK_PYTHON environment variable to your Python executable path before creating the spark instance:

```
import sys
import os
from pyspark.sql import SparkSession
os.environ["PYSPARK_PYTHON"] = sys.executable
spark = SparkSession.builder.master("local[*]").getOrCreate()
```
