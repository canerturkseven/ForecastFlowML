ForecastFlowML Docs 
====================

ForecastFlowML is a scalable machine learning forecasting framework that enables parallel training (by distributing models rather than data) of scikit-learn like models based on PySpark.

With ForecastFlowMl, you can build scikit-learn like regressors as direct multi-step forecasters, and train a seperate model for each group in your dataset.
Our package leverages the power of PySpark to efficiently handle large datasets and enables distributed computing for faster model training.

Features
--------

ForecastFlowML provides a range of features that make it a powerful and flexible tool for time-series forecasting, including:

- Works with Pandas and Pyspark DataFrames.
- Distributed model training per group in the PySpark/Pandas DataFrames.
- Direct multi-step forecasting.
- Built-in time based cross-validation,
- Extensive time-series feature engineering (lag, rollin.g mean/std, stockout, history length).
- Hyperparameter tuning for each group model with grid search.
- Supports ``scikit-learn`` like libraries such as ``LightGBM`` or ``XGBoost``.

Whether you're new to time-series forecasting or an experienced data scientist, ForecastFlowML can help you build and deploy accurate forecasting models at scale.


Installation
------------

You can install the packaging using the following command.

::

    pip install forecastflowml


.. toctree::
   :maxdepth: 1
   :hidden:

   get_started
   user_guide
   api_reference