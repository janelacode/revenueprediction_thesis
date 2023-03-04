import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

pd.options.plotting.backend = "matplotlib"

color_pal= sns.color_palette()
plt.style.use('fivethirtyeight')

def createTimeSeriesDf(filename, make_date_buckets = False):
    df=pd.read_excel(filename, parse_dates=True)
    # convert string-like month data to date
    df["Date"]= pd.to_datetime(df["Date"], format='%b_%Y')
    df = df.set_index('Date')
    if make_date_buckets:
        df['quarter']= df.index.quarter
        df['month']=df.index.month

    return df

# modifies the dataframe to insert lags and returns the name of the 
# lagged columns it inserted (as a list of names)
def insertLags(df, laggedColName = "Sales", numLags = 3):
    laggedNames = []
    
    for i in range(1, numLags + 1):
        # create the name of the lagged column
        lagName = f"{laggedColName}_lag_{i}"
        # add the name of the lagged column to the list
        laggedNames.append(lagName)
        # fill the lagged column by shifting sales i times
        df[lagName] = df[laggedColName].shift(i)
    # because of the shift, the lagged columns will contain NaNs
    # at the beginning. I drop them
    df.dropna(inplace = True)
    return laggedNames

def prepareTrainData(df, colsSelected):
    X = df[colsSelected].to_numpy()
    y = df["Sales"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
    return X_train, X_test, y_train, y_test

def trainAndFit(model, X_train, X_test, y_train, y_test, modelName):
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    fit = model.predict(X_train)
    if y_test is not None:
        rmse_rf=sqrt(mean_squared_error(pred, y_test))
        print (f"Mean Squared Error for {modelName} Model is: {rmse_rf}")
    return fit, pred

def plotPredAndTest(fit, pred, actual, label):
    plt.rcParams["figure.figsize"]=(6,3)
    plt.plot(fit,label=f"{label} (train)")
    plt.plot(range(len(fit)-1, len(fit) + len(pred)),list([fit[-1]])+list(pred),label=f"{label} (test)")

    plt.plot(actual, label=f"Actual")
    plt.legend(loc="upper left")

def makeNextTest(df, colsSelected, make_date_buckets = False):
    cols = ["Sales"] + colsSelected + ['quarter', 'month']
    lastRow = df[cols].tail(1)
    df_shifted = lastRow.shift(axis="columns")
    df_shifted.index += pd.DateOffset(months=1)
    df_shifted['quarter'] = df_shifted.index.quarter
    df_shifted['month'] = df_shifted.index.month
    
    if make_date_buckets:
        nextTest = df_shifted[colsSelected+ ['quarter', 'month']]
    else:
        nextTest = df_shifted[colsSelected]

    return nextTest

