What is ForecastFlowML?
***********************


Distributed Models Over Distributed Data
========================================

The distributed models over distributed data approach is a machine learning technique 
that involves training multiple models in parallel over subsets of a large dataset 
that are distributed across different computing nodes. This approach is commonly used
in big data applications where the dataset is too large to fit in the memory of a 
single machine, and the computations required for training the models are 
computationally intensive.

.. image:: /_static/distributed_models.svg
   :alt: Distributed Models
   :align: center


ForecastFlowML is built on this principal. ForecastFlowML enables the parallel 
processing of large datasets, allowing for faster training times and greater 
scalability. It provides a way to handle big data in a distributed computing 
environment.


Improved Accuracy
=================

Different subsets of the data can have different characteristics, including different 
distributions, feature correlations, and relationships between input features and 
output variables. These differences can make it difficult to build a single 
machine learning model that is optimal for the entire dataset.

Additionally, different subsets of the data may have varying levels of noise or
uncertainty, or may be subject to different characteristics that affect
the outcome. By using different models on different subsets of the data, distributed 
models can adapt to these differences and provide more accurate predictions.


Machine Learning Forecasting
============================

ForecastFlowML trains an independent model for forecast horizon, which is the number of
time steps into the future that we want to predict. In this approach, each model
specializes in predicting different forecast horizon. 

.. image:: /_static/direct_forecast.svg
   :alt: Image description
   :align: center

Feature Engineering
===================

ForecastFlowML has two main modules, namely preprocessing and modelling.

The preprocessing module is designed perform scalable time series feature engineering
based on PySpark. ``FeatureExtractor`` takes PySpark DataFrame as input and adds
various lag, lag-window, trend, and date features to the data. These features are added 
to capture temporal patterns and trends in the data, which can be useful for 
making accurate forecasts. Once the preprocessing module has created the necessary 
features, the features dataset is passed to the modelling module of ForecastFlowML. 

.. image:: /_static/modules.svg
   :alt: Image description
   :align: center


Training 
========

In the training phase, the dataset is divided into subsets (groups) based on the 
parameter spesified by the user and models are trained in parallel utilizing the
``Pandas UDFs``. 

``Pandas UDFs`` distributes each subset of dataset to cluster executors and converts 
data into a ``Pandas DataFrame``. In this way, we can still utilize Python based
machine learning algorithms which have ``scikit-learn`` like API. 

.. warning::
   ``Pandas UDFs`` expect each subset (group) in your dataframe to fit in the memory 
   of a single machine. Otherwise, you might face out-of-memory problems for 
   large sized subsets.

.. image:: /_static/train.svg
   :alt: Image description
   :align: center

After distributing the slices of dataframe, the training phase starts. 
As mentioned above, ``FeatureExtractor`` creates various lag features such as 
``lag_1``, ``lag_2`` and ``lag_3``. Since, we are training models for different
forecast horizons, some models do not have access to spesific lags as they will
be unknown in the inference phase.

For instance:

- ``Step+1`` model is allowed to use all lags: ``lag_1``, ``lag_2`` and ``lag_3``.
- ``Step+2`` model is allowed to use ``lag_2``, ``lag_3``; but not ``lag_1``.
- ``Step+3`` model is allowed to use ``lag_3``; but not ``lag_2`` and ``lag_3``.

After the not allowed lag features are removed, an independent model for each forecast
horizon is trained with other features. 


Inference
=========

Similar to training phase, for inference the data is also is divided into 
subsets (groups). Then, trained models generate forecast for each horizon 
in a parallel way utilizing the ``Pandas UDFs``. 

.. image:: /_static/predict.svg
   :alt: Image description
   :align: center




