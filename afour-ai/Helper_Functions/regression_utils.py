from xmlrpc.client import Boolean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import math
import seaborn as sns
from keras import callbacks
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, chi2
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import pickle


# Handling Null Values

# Iterative Imputer
def iterative_imputer(df: pd.DataFrame, imputation_order = ascending, initial_strategy = 'mean'):
    columns = df.columns()
    imputer = IterativeImputer(imputation_order = imputation_order, initial_strategy = initial_strategy)  # define imputer
    imputer.fit(df)  # fit on the dataset
    data_trans = imputer.transform(df)  # transform the dataset
    df_trans = pd.DataFrame(data_trans, columns = columns)
    return df_trans

# Finding the number and percentage of null values in a column
def count_null_perc(df : pd.DataFrame):
    for i in range(df.shape[1]):
    # count number of rows with missing values
        n_miss = df[[i]].isnull().sum()
        perc = n_miss / df.shape[0] * 100
        print('> %d, Missing: %d (%.1f%%)' % (i, n_miss, perc))


# Plotting Heatmap Correlation plot to find highest correlation features for the label

# it takes the whole dataframe as an input
def plot_heatmap(df):
    corr = df.corr()
    dataplot = sns.heatmap(corr, cmap="YlGnBu", annot=True)
    plt.show()
    
# Plotting the Histogram for the given list/dataframe

def plot_hist(x, bins:int = 100):
    plt.hist(x, bins = bins)
    plt.xlabel('X-Axis')
    plt.ylabel('Y-Axis')
    plt.show()

# Splitting Data

# Splitting the data based on the frac list given
# fracs = [0.4,0.3,0.2,0.1] will give us 4 different datasets with 40%, 30%, 20% and 10% of the orignal dataset
# it takes the DataFrame, list of fraction whose sum must be 1 and an int for random state as input
def split_by_fractions(df:pd.DataFrame, fracs:list, random_state:int=42):
    assert sum(fracs)==1.0, 'fractions sum is not 1.0 (fractions_sum={})'.format(sum(fracs))
    remain = df.index.copy().to_frame()
    res = []
    for i in range(len(fracs)):
        fractions_sum=sum(fracs[i:])
        frac = fracs[i]/fractions_sum
        idxs = remain.sample(frac=frac, random_state=random_state).index
        remain=remain.drop(idxs)
        res.append(idxs)
    return [df.loc[idxs] for idxs in res]

# Splitting Dataset into training, validation and test dataset
# it takes the dataframe, label column name and test size as main input
def regressionsplit_train_test(df: pd.DataFrame, label:str, random:int = None, test_size:int = 0.25, shuffle:Boolean = True):
  y = df[label].copy()
  X = df.drop([label], 1)
  X_train, y_train, X_test, y_test = train_test_split(X, y,test_size = test_size, random_state = random, shuffle = shuffle)
  return X_train, y_train, X_test, y_test

# Saving and Loading of model to keep the model with highest accuracy and use it for later purpose
def save_model(model,file_loc):
  """
  Saves the model to given file location as a pickle file
  """
  pickle.dump(model, open(file_loc, 'wb'))

def load_model(file_loc):
  """
  Returns the loaded pickle file from given file location
  """
  return pickle.load(open(file_loc, 'rb'))

# SelectKBest Features

# it selects the best 'k' features from the available features
# it takes dataframe, label column name, score func(f_classif or f_regression), k value for both categorical and numerical columns
def SelectKBest_features(df:pd.DataFrame, label:str, score_func = f_classif, num_k: int = 5, cat_k: int = 0):
    y = df[label].copy()  #label column
    X = df.drop([label], 1) #feature columns
    numerical_ix = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_ix = X.select_dtypes(include=['object', 'bool']).columns
    if len(numerical_ix) != 0:
        # numerical_ix_new = SelectKBest(score_func = score_func, k = num_k).fit(X[numerical_ix],y)
        num_new = SelectKBest(score_func = score_func, k = num_k).fit_transform(X[numerical_ix],y)
    elif len(categorical_ix != 0):
        # categorical_ix_new = SelectKBest(score_func = chi2, k = num_k).fit(X[categorical_ix],y)
        cat_new = SelectKBest(score_func = chi2, k = cat_k).fit_transform(X[categorical_ix],y)
    if len(numerical_ix) != 0 and len(categorical_ix) != 0:
        X_new = pd.DataFrame(num_new,cat_new)
    elif len(numerical_ix) != 0 and len(categorical_ix) == 0:
        X_new = pd.DataFrame(num_new)
    elif len(numerical_ix) != 0 and len(categorical_ix) != 0:
        X_new = pd.DataFrame(cat_new)
    return X_new,y

# Dealing with Categorical Columns

# converting the categorical values into unique values using label encoder
def get_labelencoder(x):
    label_encoder = preprocessing.LabelEncoder()
    return label_encoder.fit_transform(x)

# coverting the unique catogerical values in a column into a binary column using onehot encoder
def get_onehotencoder(df, cat_col):
    df = pd.get_dummies(data=df, drop_first=True, columns = [[cat_col]])
    return df

# Finding the optimal number of epochs so as to not waste time using EarlStopping callback function
def earlystop():
    earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 5, 
                                        restore_best_weights = True)
    return earlystopping

# Finding the Evaluation Metrics such to count accuracy and loss
def accuracy_param(y_test, y_pred):
  mse = mean_squared_error(y_test, y_pred)   #Mean Square Error(mse)
  mae = mean_absolute_error(y_test, y_pred)  #Mean Absolute Error(mae)
  rmse = math.sqrt(mean_squared_error(y_test, y_pred))   #Root Mean Square Error(rmse)
  mape = np.mean(np.abs(y_pred - y_test)/np.abs(y_test)) #Mean Absolute Percentage Error(mape)
  mase = np.mean(np.abs(y_pred - y_test)/ mae) #Mean Absolute Scaled Error
  print('MSE: '+str(mse))
  print('MAE: '+str(mae))
  print('RMSE: '+str(rmse))
  print('MAPE: '+str(mape))
  print('MASE: '+str(mase))