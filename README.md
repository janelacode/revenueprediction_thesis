# Predict return on investment of a medical manufacturing company

My master's thesis is currently in progress.
I am predicting future sales 1-3 years in advance based on historical sales data.
My analysis starts with a statistical outlook using naive models (e.g moving averages) and then uses more complex ML models (XGBoost, Prophet) to try improving on those results both in terms of trend and seasonality.

To do so, I try to combine time series forecast with external regressors that are easier to predict. This way, I can 'ground' my prediction (limit uncertainty explosion). One of my most hopeful external regressors so far are flu vaccination rates.

My latest progress is in [this jupyter notebook](https://github.com/janelacode/revenueprediction_thesis/blob/main/prophet.ipynb) achieving 20-40% improvement with ML.

An analysis of my predictive performance is in [this notebook](https://github.com/janelacode/revenueprediction_thesis/blob/main/compare_sarima_prophet.ipynb)

