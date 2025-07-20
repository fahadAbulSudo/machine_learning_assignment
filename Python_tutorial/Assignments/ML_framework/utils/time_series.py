import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import math
import seaborn as sns
from keras import callbacks
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from seaborn.palettes import QUAL_PALETTE_SIZES
from statsmodels.tsa.stattools import kpss, adfuller


# Splitting Data into Training, Validation and Test dataset

# Splitting the data further into X(features) and y(label)
def split_X_Y(train, valid, test, label):
    y_train = train[label].copy()
    X_train = train.drop([label], 1)

    y_valid = valid[label].copy()
    X_valid = valid.drop([label], 1)

    y_test  = test[label].copy()
    X_test  = test.drop([label], 1)
    return X_train, y_train, X_valid, y_valid, X_test, y_test

# Splits the orignal dataset into train, vlid and test dataset and if only the algorithm is XGBoost it goes to above function
def split_train_test_validation(df, test_size, valid_size, label, arima = False):
    test_split  = int(df.shape[0] * (1-test_size))
    valid_split = int(df.shape[0] * (1-(valid_size+test_size)))

    train  = df[:valid_split].copy()
    valid  = df[valid_split+1:test_split].copy()
    test  = df[test_split+1:].copy()
    if arima == False:
        return split_X_Y(train, valid, test, label)
    else:
        return train.values, valid.values, test.values

# Split train and test
def split_train_test(df, test_size):
    test_split  = int(df.shape[0] * (1-test_size))
    train  = df[:test_split].copy()
    test  = df[test_split+1:].copy()
    # train = train.reset_index()
    # test = test.reset_index()
    return train, test

    
# Plotting Data

# PLotting the Label Feature
def plot_target(df, label):    #label here is the column whose values are to be predicted
  plt.figure(figsize=(10,6))
  plt.grid(True)
  plt.xlabel('Date')
  plt.ylabel('Target Column')
  plt.plot(df[label])
#   plt.title('A closing price')
  plt.show()
  
# Plotting the decompose plot of the feature column


# Plotting the data as training, validation, testing and predicted data in different colour
# y_train, y_valid and y_test must be a DataFrame with date as Index or 
# should be converted into one before passing through the function
def plotting_dataNforecast(y_train: pd.DataFrame, y_valid: pd.DataFrame, y_test: pd.DataFrame, y_pred):
  fc = pd.DataFrame(y_pred)  # Converting the forecast value array into a DataFrame
  fc.index = y_test.index

  plt.figure(figsize=(10,5), dpi=100)
  plt.plot(y_train, color = 'black', label='Training Data')
  plt.plot(y_valid, color = 'black', label='Validation Data')
  plt.plot(y_test, color = 'orange', label='Actual Data')
  plt.plot(fc, color = 'yellow',label='Predicted Data')

  plt.title('Test Data Prediction')
  plt.xlabel('Time')
  plt.ylabel('Label')
  plt.legend(loc='upper left', fontsize=8)
  plt.show()


#Finding the optimum value of p,d and q such that we have the best possible model for forecasting
def optimum_pdq(train, max_p = 5, max_q = 5):
  model_autoARIMA = auto_arima(train, start_p=0, start_q=0,
                        test='adf',       # use adftest to find optimal 'd'
                        max_p=max_p, max_q=max_p, # maximum p and q
                        m=1,              # frequency of series
                        d=None,           # let model determine 'd'
                        seasonal=False,   # No Seasonality
                        start_P=0, 
                        D=0, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
  print(model_autoARIMA.summary())
  model_autoARIMA.plot_diagnostics(figsize=(15,8))
  plt.show()


# Shows a plot where the the data is decomposed such that we can see the Seasonality and Trend in the data
def plot_decompose(df:pd.DataFrame, period:int, index:str, label:str):
  df_dec = df[[index, label]].copy()
  df_dec = df_dec.set_index(index)
  decomp = seasonal_decompose(df_dec, model='additive', period=period)
  fig = decomp.plot()
  fig.set_size_inches(20, 8)

# ADF Testing
def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")

#KPSS Testing
def kpss_test(timeseries):
    print("Results of KPSS Test:")
    kpsstest = kpss(timeseries, regression="c")
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    print(kpss_output)

# LSTM Functions

def df_to_x_y(df, label_index:int, window_size=11):
  df_as_np = df.to_numpy()
  X = []
  Y = []
  for i in range(len(df_as_np)-window_size):
    row = [a for a in df_as_np[i:i+window_size]]
    X.append(row)
    label = df_as_np[i+window_size][label_index]
    Y.append(label)
  return np.array(X), np.array(Y)


def split_X_Y(train, valid, test, label):
    y_train = train[label].copy()
    X_train = train.drop([label], 1)

    y_valid = valid[label].copy()
    X_valid = valid.drop([label], 1)

    y_test  = test[label].copy()
    X_test  = test.drop([label], 1)
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def split_train_test_validation(df, test_size, valid_size, label, arima = False):
    test_split  = int(df.shape[0] * (1-test_size))
    valid_split = int(df.shape[0] * (1-(valid_size+test_size)))

    train  = df[:valid_split].copy()
    valid  = df[valid_split+1:test_split].copy()
    test  = df[test_split+1:].copy()
    if arima == False:
        return split_X_Y(train, valid, test, label)
    else:
        return train.values, valid.values, test.values

def createXY(dataset, target, start, end, window, horizon):
    X = []
    y = []
    start = start + window
    if end is None:
        end = len(dataset) - horizon#here try to give end as none because this line of code prevent from index error

    for i in range(start, end):
        indices = range(i-window, i)
        X.append(dataset[indices])

        indicey = range(i+1, i+1+horizon)
        y.append(target[indicey])
    return np.array(X), np.array(y)

def split_train_test_LSTM(X, y, valid_size, test_size):
  test_split  = int(len(X) * (1-test_size))
  valid_split = int(len(X) * (1-(valid_size+test_size)))

  X_train  = X[:valid_split].copy()
  X_valid  = X[valid_split+1:test_split].copy()
  X_test  = X[test_split+1:].copy()

  y_train  = y[:valid_split].copy()
  y_valid  = y[valid_split+1:test_split].copy()
  y_test  = y[test_split+1:].copy()

  return X_train, y_train, X_valid, y_valid, X_test, y_test