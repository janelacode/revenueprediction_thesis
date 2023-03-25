from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
import math
import pickle
from sklearn.metrics import mean_squared_error
from numpy import array
import pandas as pd
from pandas import concat
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def is_cold(el):
    return (el.month > 10 or el.month < 3)

def prepare_time_series(df, original_df = None):
    df['is_cold'] = df['ds'].apply(is_cold)
    df['is_not_cold'] = ~df['ds'].apply(is_cold)
    if original_df is not None:
        original_df_restricted = original_df.loc[df.index.min():df.index.max(),:]
        return pd.concat([df, original_df_restricted], axis=1)
        
    return df

def create_and_fit_model(
    train_df, 
    quarterly_period=0, 
    quarterly_fourier_order=5, 
    iscold_fourier_order=1,
    fourier_order_isnotcold=2,
    changepoint_prior_scale=1., 
    yearly_seasonality=5,
    iscold_period=4,
    isnotcold_period=8,
    vaccine_period = 0,
    cpi_period = 0,
    vaccine_prior_scale = 0,
    cpi_prior_scale = 0,
    vaccine_method = 'multiplicative',
    cpi_method = 'addtive'
):
    model = Prophet(daily_seasonality=False, weekly_seasonality=False, changepoint_prior_scale=changepoint_prior_scale, yearly_seasonality=yearly_seasonality)
    if vaccine_period > 0:
        model.add_regressor('vaccine', prior_scale=vaccine_prior_scale, mode=vaccine_method)
    if cpi_period > 0:
        model.add_regressor('CPI', prior_scale=cpi_prior_scale, mode=cpi_method)

    if quarterly_period > 0:
        model.add_seasonality(name='quarterly', period=quarterly_period, fourier_order=quarterly_fourier_order)
    if iscold_period > 0:
        model.add_seasonality(name='is_cold', period=iscold_period, fourier_order=iscold_fourier_order, prior_scale = 2, condition_name='is_cold')
    if isnotcold_period > 0:
        model.add_seasonality(name='is_not_cold', period=isnotcold_period, fourier_order=fourier_order_isnotcold, prior_scale = 2, condition_name='is_not_cold')
    model.fit(train_df)
    return model

def print_mserr(df_gt, df_pred, naive_pred, best = None):
    df_pred = df_pred[-len(df_gt):]
    error1 = mean_squared_error(df_gt[['y']].values, df_pred[['yhat']].values)
    error2 = mean_squared_error(df_gt[['y']].values, [naive_pred
    ]*len(df_gt))
    perc_improvement = (math.sqrt(error2)-math.sqrt(error1))/max(math.sqrt(error1),math.sqrt(error2))
    return math.sqrt(error1)

def configure_and_fit(col, external_name, df_sales, df_vaccines, df_cpi, kwargs):
        df = pd.concat([df_sales, df_vaccines, df_cpi], axis=1)
        df['ds'] = df.index
        df['y']=df[col]
        df = prepare_time_series(df)
        train_df = df[:-cutoff_train_test]
        test_df = df[-cutoff_train_test:]
        print(f"prediction for {col}")

        model = create_and_fit_model(train_df, **kwargs)

        future = model.make_future_dataframe(periods=24, freq = 'MS')
        future['Date'] = future['ds']
        future.set_index('Date', inplace=True)
        future = prepare_time_series(future, df[[external_name]])

        forecast = model.predict(future)
        forecast['Date'] = forecast['ds']
        forecast.set_index('Date', inplace=True)

        viz_df = train_df.join(forecast[['yhat', 'yhat_lower','yhat_upper']], how = 'outer')
        return print_mserr(test_df, forecast, train_df[-12:]['y'].mean(), 1e99)


from prophet import Prophet
from prophet.plot import add_changepoints_to_plot

import math

cutoff_train_test = 24
df_sales = pd.read_excel("data_for_clustering.xlsx", parse_dates=True,usecols = "A,F,G,I,J,K,L,M").set_index('Date')
df_sales.index = pd.to_datetime(df_sales.index)
df_vaccines = pd.read_csv("vaccines.csv", parse_dates=True)

df_vaccines['Date'] = pd.to_datetime(df_vaccines['Date'], format='%Y-%m')
# df_vaccines['vaccine'] = df_vaccines['vaccine'].shift(-1)

# restrict additional regressor to same time
df_vaccines = df_vaccines.set_index('Date')

# restrict additional regressor to same time
df_vaccines = df_vaccines.loc[df_sales.index.min():df_sales.index.max(),:]

cpi_cz = [99.5, 99.6, 99.6, 99.6, 99.7, 99.7, 100.0, 99.9, 99.6, 99.8, 99.5, 99.5, 99.7, 99.5, 99.7, 99.8, 100.1, 100.4, 100.5, 100.4, 100.2, 100.0, 100.0, 99.6, 99.5, 100.0, 100.1, 100.2, 100.1, 100.7, 100.5, 100.6, 100.9, 100.8, 100.5, 100.8, 101.2, 101.5, 100.7, 102.3, 102.7, 102.7, 102.7, 102.9, 102.9, 103.4, 103.3, 103.2, 103.7, 103.8, 103.9, 103.1, 104.5, 104.5, 104.4, 104.7, 105.2, 105.6, 105.8, 105.9, 105.6, 106.0, 105.9, 106.0, 105.3, 107.1, 107.3, 107.5, 107.6, 108.3, 108.5, 108.9, 109.0, 108.4, 108.9, 109.2, 109.4, 108.3, 111.0, 111.3, 111.2, 111.0, 111.4, 112.1, 112.6, 112.6, 111.9, 112.1, 112.1, 111.9, 111.8, 113.4, 113.6, 113.8, 114.4, 114.6, 115.2, 116.4, 117.2, 117.4, 118.6, 118.8, 119.3, 116.1, 124.6, 126.2, 128.3, 130.6, 132.9, 135.0, 136.8, 137.4, 138.5, 136.5, 138.1, 138.1, 133.6, 146.4, 147.3]
data_cpi = {'Date': pd.date_range(start='2014-01-01', periods=len(cpi_cz), freq='MS'),
        'CPI': cpi_cz}

df_cpi = pd.DataFrame(data_cpi).set_index('Date')
df_cpi = df_cpi.loc[df_sales.index.min():df_sales.index.max(),:]
kwargs = {
     'changepoint_prior_scale': 0.2, 
     'fourier_order_isnotcold': 1, 
     'iscold_fourier_order': 5, 
     'iscold_period': 1, 
     'isnotcold_period': 1, 
     'quarterly_fourier_order': 1, 
     'quarterly_period': 4, 
     'yearly_seasonality': 4, 
     'vaccine_method': 'multiplicative', 
     'vaccine_period': 0, 
     'vaccine_prior_scale': 1,
     'cpi_prior_scale': 1,
     'cpi_method': 'multiplicative'
} 

col_categories = {'cpi':[], 'vaccine': []}
for col in df_sales.columns:
        kwargs['cpi_period'] = 0
        kwargs['vaccine_period'] = 0
        err_wo_vaccine = configure_and_fit(col, 'vaccine', df_sales, df_vaccines, df_cpi, kwargs)
        kwargs['vaccine_period'] = 1
        err_w_vaccine = configure_and_fit(col, 'vaccine', df_sales, df_vaccines, df_cpi, kwargs)
        if err_w_vaccine < err_wo_vaccine:
                col_categories['vaccine'].append((col,err_w_vaccine - err_wo_vaccine))
        
        kwargs['vaccine_period'] = 0
        kwargs['cpi_period'] = 0
        err_wo_vaccine = configure_and_fit(col, 'CPI', df_sales, df_vaccines, df_cpi, kwargs)
        kwargs['cpi_period'] = 1
        err_w_vaccine = configure_and_fit(col, 'CPI', df_sales, df_vaccines, df_cpi, kwargs)
        if err_w_vaccine < err_wo_vaccine:
                col_categories['cpi'].append((col,err_w_vaccine - err_wo_vaccine))

for key, value in col_categories.items():
    print(f"The following components have influance on {key}")
    for (name, delta) in value:
         print(f"\t* {name} with RMSE delta {delta}")