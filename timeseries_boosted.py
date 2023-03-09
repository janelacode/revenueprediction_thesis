import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.metrics import mean_squared_error
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

def createTimeSeriesDf(filename):
    df=pd.read_excel(filename, parse_dates=True)
    # convert string-like month data to date
    df["Date"]= pd.to_datetime(df["Date"], format='%b_%Y')
    df = df.set_index('Date')
    df['quarter']= df.index.quarter
    df['month']=df.index.month
    df['year']=df.index.year

    return df.astype({'month' : 'int32', 'quarter': 'int32'})

def createTimeSeriesInflationDf(filename):
    if filename[-3:] == 'csv':
        df=pd.read_csv(filename, parse_dates=True)
    else:
        df=pd.read_excel(filename, parse_dates=True)
    # convert string-like month data to date
    df["Date"]= pd.to_datetime(df["time"])
    df.rename(columns={'value': 'inflationCZK'}, inplace=True)

    # interpolate
    df = df.set_index('Date')
    df = df.resample('M').mean().interpolate()
    new_index = df.index + pd.offsets.MonthBegin(0, normalize=True)
    df = df.set_index(new_index)

    df['quarter']= df.index.quarter
    df['month']=df.index.month
    df['year']=df.index.year

    return df

def plotDataSummary(df, colname = 'Sales'):
    # Create the first plot with full width
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
    # Create the first plot with full width
    ax1 = axs[1].inset_axes([0, 0.5, 0.5, 0.5])
    ax2 = axs[1].inset_axes([0.5, 0.5, 0.5, 0.5])


    df[colname].plot(x='Date', y=colname, style='-', color=color_pal[0], title=f'{colname} as a relation of time', ax=axs[0])

    df[colname].plot(color=color_pal[1], title=f"Frequency of {colname}", ax=ax1, kind='hist', bins=10)
    ax2.set_title(f"{colname} per month")
    sns.boxplot(data=df, x='month', y=colname, ax=ax2)
    fig.tight_layout()

def addLags(df):
    target_map = df['Sales'].to_dict()
    df['lag1'] = (df.index - pd.DateOffset(months=4)).map(target_map)
    df['lag2'] = (df.index - pd.DateOffset(months=5)).map(target_map)
    df['lag3'] = (df.index - pd.DateOffset(months=6)).map(target_map)
    return df
