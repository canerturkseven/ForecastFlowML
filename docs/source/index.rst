ForecastFlowML Docs 
====================

ForecastFlowML is a scalable machine learning forecasting framework that enables parallel training (by distributing models rather than data) of scikit-learn like models based on PySpark.

With ForecastFlowML, you can build scikit-learn like regressors as direct multi-step forecasters, and train a seperate model for each group in your dataset.
Our package leverages the power of PySpark to efficiently handle large datasets and enables distributed computing for faster model training.

Features
--------

ForecastFlowML provides a range of features that make it a powerful and flexible tool for time-series forecasting, including:

- Scaleable and extensive time series feature engineering (lag, rolling mean/std, stockout, history length) with PySpark.
- Parallel model training per group in the dataset with Pyspark Pandas UDFs.
- Direct multi-step forecasting.
- Built-in time based cross-validation.
- Hyperparameter tuning for each group model with grid search.
- Supports `scikit-learn` like libraries such as `LightGBM` or `XGBoost`.

Installation
------------

ForecastFlowML installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~
You can install the package using the following command:

.. code-block:: console

   pip install forecastflowml

Check Java
~~~~~~~~~~
Make sure you have installed Java 11. You can check whether you have Java or not with the following command:

.. code-block:: console

   java -version

Set PYSPARK_PYTHON
~~~~~~~~~~~~~~~~~~
In the python script, set PYSPARK_PYTHON environment variable to your Python executable path before creating the spark instance:

.. code-block:: python

   import sys
   import os
   from pyspark.sql import SparkSession
   os.environ["PYSPARK_PYTHON"] = sys.executable
   spark = SparkSession.builder.master("local[*]").getOrCreate()

.. toctree::
   :maxdepth: 1
   :hidden:

   user_guides
   examples
   api_reference