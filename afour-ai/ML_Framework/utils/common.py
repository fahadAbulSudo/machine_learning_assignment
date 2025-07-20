from __future__ import absolute_import, division, print_function, unicode_literals
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sl
import tensorflow as tf
import configparser
import zipfile
from pathlib import Path
import os
from tqdm import tqdm

from sklearn import preprocessing
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow.keras.callbacks import TensorBoard
from time import time
from sklearn.metrics import explained_variance_score,mean_squared_error, r2_score,max_error, mean_absolute_error, mean_absolute_percentage_error

# Function to get the dataset information
def get_dataset_info(df, show_description=False, show_columns=False):
  """
  This function provides all the information
  of the given dataset
  Args:
    df: Dataset
  """
  print("="*100)
  print("Head:")
  print(df.head())
  print("="*100)
  print(f"Shape: {df.shape}")
  print("="*100)
  if show_description:
    print("Description:")
    print(df.describe())
    print("="*100)
  if show_columns:
    print("Columns:")
    print(df.columns)
    print("="*100)


# Function to create tensorboard callback to keep track of model's history so we can inspect it later
def create_tensorboard_callback(dir_name, experiment_name):
  """
  Creates a TensorBoard callback instand to store log files.
  Stores log files with the filepath:
    "dir_name/experiment_name/current_datetime/"

  Args:
    dir_name: target directory to store TensorBoard log files
    experiment_name: name of experiment directory (e.g. efficientnet_model_1)
  """
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = TensorBoard(log_dir=log_dir)
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback


# Function to plot the graph with given parameters
def plot_graph(valueX, valueY, title=None, start=None, end=None):
  """
  This function takes the features as arguments and plot them.

  Args:
    valueX: First value. This should be a feature in the dataframe
    valueY: Second value. This should be a feature in the dataframe
    title: title of the graph
    start: starting index
    end: end index
  """
  plt.plot(valueX[start:end], 'g', label=f'{valueX}')
  plt.plot(valueY[start:end], 'b', label=f'{valueY}')
  plt.title(title)
  plt.xlabel(f'{valueX}')
  plt.ylabel(f'{valueY}')
  plt.legend()
  plt.show()


# Function to find the count and percentage of missing values in each column
def get_count_and_percentage_missing_values(df):
  """
  This function prints the count and percentage of missing values
  for each column in the dataframe
  Args:
    df = dataframe
  """
  counts = df.isna().sum()
  percentages = round(df.isna().mean() * 100, 1)
  null_values = pd.concat([counts, percentages], axis=1, keys=["count", "% null"])
  print(null_values)


# Function to replace missing values in the dataset using SimpleImputer
# Ref: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
def replace_missing_values(df, strategy='mean'):
  """
  This function replaces the missing/nan values
  by using sklearn SimpleImputer

  Args:
    df: dataframe
    strategy: The imputation strategy (mean/median/most_frequent/constant)
  """
  # Save the columns in the dataframe
  column_list = df.columns.to_list()
  imputer = SimpleImputer(missing_values = np.nan,
                          strategy = strategy)
  imputer = imputer.fit(df)
  # Imputing the data    
  data = imputer.transform(df)
  # Convert data (ndarray) into the dataframe
  df = pd.DataFrame(data, columns = column_list)
  return df


# This function shows the execution time of 
# the function object passed
def timer_func(func):
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func


# Returns reduced Dimensions using PCA given data frame and resultant number of dimensions needed
def reduce_dim_PCA(df, n=3):
  default_col_names = ["col"+str(i+1) for i in range(n)]
  pca = PCA(n_components=n)
  pca.fit(df)
  PCA_ds = pd.DataFrame(pca.transform(df), columns=(default_col_names))
  return PCA_ds

def iqr(df, val):
    Q1 = df[val].quantile(0.05)
    Q3 = df[val].quantile(0.95)
    IQR = Q3 - Q1
    return df[(df[val]>= Q1 - 1.5*IQR) & (df[val] <= Q3 + 1.5*IQR)]

# Given dataframe, function will determine which columns should be normalised or scaled based on skewness
# Return to be scaled column list , normalised column list
def preprocess_numeric_column_data(data):
  for i in data.columns:
    if data.dtypes[i] == np.object:
      print("Cannot process object data")
      return

  ## if skew is 0.5 and -0.5 i.e its normal distribution and use scaler else normalise
  scale_list = []
  normalise_list = []
  data_skew = data.skew()

  for i in data.columns:
    if data_skew[i] >= -0.5 and data_skew[i] <= 0.5:
      scale_list.append(i)
    else:
      normalise_list.append(i)
  return (scale_list, normalise_list)


# Visualise categorical via countplot and numeric data via scatterplot
def visualise_data(df, x_value, y_value, typeofdata="numeric", count_value = "count_value"):
  if typeofdata == "categorical":
    df.x_value.value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))
    plt.title(f'Number of {count_value} by {x_value}')
    plt.ylabel(f'Number of {count_value}')
    plt.xlabel(f'{x_value}')
  elif typeofdata == "numeric":
    plt.figure(figsize=(20,8))
    g=sns.scatterplot(x=x_value,y=y_value,data=df)
    g.set_title(f'{x_value} vs {y_value} Correlation',fontsize=20)
    g.set_xlabel(f'{x_value}',fontsize=10)
    g.set_ylabel(f'{y_value}',fontsize=10)
  else:
    print("Type of data passed isn't supported!!")
    return


def column_correlation_importance(df, col_name):
  corrmat= df.corr()
  print(corrmat[col_name].abs().sort_values(ascending=False))

# Handling Null values
"""
initial_strategy : mean, median, most-frequent and constant
imputation_order : ascending, descending, roman(left to right), arabic(right to left), random
estimator object, default=BayesianRidge()
"""
def iterative_imputer(df: pd.DataFrame, imputation_order = 'ascending', initial_strategy = 'mean'):
  columns = df.columns()
  imputer = IterativeImputer(imputation_order = imputation_order, initial_strategy = initial_strategy)  # define imputer
  imputer.fit(df)  # fit on the dataset
  data_trans = imputer.transform(df)  # transform the dataset
  df_trans = pd.DataFrame(data_trans, columns = columns)
  return df_trans

def count_null_perc(df : pd.DataFrame):
  for i in range(df.shape[1]):
    # count number of rows with missing values
    n_miss = df[[i]].isnull().sum()
    perc = n_miss / df.shape[0] * 100
    print('> %d, Missing: %d (%.1f%%)' % (i, n_miss, perc))

# Plotting Heatmap
def plot_heatmap(df):
  corr = df.corr()
  print(corr)
  dataplot = sns.heatmap(corr, cmap="YlGnBu", annot=True)
  plt.show()

# Plotting Histogram
def plot_hist(x, bins=100):
  plt.hist(x, bins = bins)
  plt.xlabel('X-Axis')
  plt.ylabel('Y-Axis')
  plt.show()

# Selecting K Best Features
def select_k_best_features(df:pd.DataFrame, label:str, score_func = f_classif, num_k: int = 5, cat_k: int = 0):
  y = df[label].copy()  #label column
  X = df.drop([label], 1) #feature columns
  numerical_ix = X.select_dtypes(include=['int64', 'float64']).columns
  categorical_ix = X.select_dtypes(include=['object', 'bool']).columns
  if len(numerical_ix) != 0:
    num_new = SelectKBest(score_func = score_func, k = num_k)
    num_new.fit(X[numerical_ix],y)
    num_col = num_new.get_support(indices=True)
  if len(categorical_ix != 0):
    cat_new = SelectKBest(score_func = chi2, k = cat_k)
    cat_new.fit(X[categorical_ix],y)
    cat_col = cat_new.get_support(indices=True)
  if len(numerical_ix) != 0 and len(categorical_ix) != 0:
    X_new = pd.DataFrame(num_new,cat_new)
  elif len(numerical_ix) != 0 and len(categorical_ix) == 0:
    # X_new = pd.DataFrame(num_new)
    X_new = df.iloc[:,num_col]
  elif len(numerical_ix) == 0 and len(categorical_ix) != 0:
    X_new = pd.DataFrame(cat_new)
  return X_new

# IMproved K-Best Features
def select_k_best_features(df:pd.DataFrame, label:str, score_func = f_classif, num_k: int = 5, cat_k: int = 0):
  y = df[label].copy()  #label column
  X = df.drop([label], 1) #feature columns
  numerical_ix = X.select_dtypes(include=['int64', 'float64']).columns
  categorical_ix = X.select_dtypes(include=['object', 'bool', 'uint8']).columns
  # print(categorical_ix)
  if len(numerical_ix) != 0:
    num_new = SelectKBest(score_func = score_func, k = num_k)
    num_new.fit(X[numerical_ix],y)
    num_col = num_new.get_support(indices=True)
  if len(categorical_ix != 0):
    cat_new = SelectKBest(score_func = chi2, k = cat_k)
    cat_new.fit(X[categorical_ix],y)
    cat_col = cat_new.get_support(indices=True)
  col = []
  if len(numerical_ix) != 0 and len(categorical_ix) != 0:
    for i in num_col:
      col.append(df.columns[i])
    for i in cat_col:
      col.append(df.columns[i])
    return col
  elif len(numerical_ix) != 0 and len(categorical_ix) == 0:
    for i in num_col:
      col.append(df.columns[i])
    return col
  elif len(numerical_ix) == 0 and len(categorical_ix) != 0:
    for i in cat_col:
      col.append(df.columns[i])
    return col

def process_normal(df, normalise_list):
    features = df[normalise_list]
    normal = preprocessing.MinMaxScaler().fit(features.values)
    features = normal.transform(features.values)
    return features, normal

# Data Cleaning
"""
Standardisation
"""
def standardization(df, column_list):
  scaler = preprocessing.StandardScaler()
  df[column_list] = scaler.fit_transform(df[column_list])

"""
Normalization
If robust = true do robust scaling 
else do min max scaling
"""
def normalization(df, column_list, robust = True):
  if robust:
    scaler = preprocessing.RobustScaler()
    df[column_list] = scaler.fit_transform(df[column_list])
  else:
    scaler = preprocessing.MinMaxScaler()
    df[column_list] = scaler.fit_transform(df[column_list])

# Dealing with Categorical Values
"""
Label Encoding
"""
def get_labelencoder(x):
  label_encoder = preprocessing.LabelEncoder()
  return label_encoder.fit_transform(x)

"""
One-Hot Encoding/Dummy Variable
"""
def get_onehotencoder(df, cat_col):
  df = pd.get_dummies(data=df, drop_first=True, columns = cat_col)
  return df

# Boxplot
"""
Pass the dataframe and the list of columns to be plotted
Eg. columns = ['col1', 'col2', 'col3']
"""
def plot_boxplot(df, columns):
  boxplot = df.boxplot(column = columns)  

# Plotting the Target Column
def plot_target(df, label):    #label here is the column whose values are to be predicted
  plt.figure(figsize=(10,6))
  plt.grid(True)
  plt.xlabel('Date')
  plt.ylabel('Target Column')
  plt.plot(df[label])
  plt.title('A closing price')
  plt.show()

# Remove outliers using IQR Technique
def remove_outlier_IQR(df):
  """
  This function takes the dataset and applies IQR technique
  to remove the outiers. 
  It returns the dataset with removed outliers
  Args:
    df: dataframe
  Return:
    df_final: final dataframe after removing the outliers
  """
  Q1=df.quantile(0.25)
  Q3=df.quantile(0.75)
  IQR=Q3-Q1
  df_final = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
  return df_final

def remove_outlier(df, col_name):
  q1 = df[col_name].quantile(0.25)
  q3 = df[col_name].quantile(0.75)
  iqr = q3-q1 #Interquartile range
  fence_low  = q1-1.5*iqr
  fence_high = q3+1.5*iqr
  df_out = df.loc[(df[col_name] > fence_low) & (df[col_name] < fence_high)]
  return df_out

def model_evaluate(model, y_test, y_pred):
  exp_var_score = explained_variance_score(y_test, y_pred)
  max_err= max_error(y_test, y_pred)
  r2= r2_score(y_test, y_pred)
  mae= mean_absolute_error(y_test, y_pred)
  mse= mean_squared_error(y_test, y_pred)
  rmse= np.sqrt(mse)
  mape = mean_absolute_percentage_error(y_test, y_pred)

  row_label = [model]
  
  data_score = { 'exp_varne': exp_var_score, 'max_error':max_err, 
                'r2': r2, 'mae':mae, 'mse':mse, 'rmse':rmse, 'mape':mape,}
  
  df_data = pd.DataFrame(data= data_score, index= row_label)
  
  return df_data

# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
  nunique = df.nunique()
  df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
  nRow, nCol = df.shape
  columnNames = list(df)
  nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
  plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
  for i in range(min(nCol, nGraphShown)):
      plt.subplot(nGraphRow, nGraphPerRow, i + 1)
      columnDf = df.iloc[:, i]
      if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
          valueCounts = columnDf.value_counts()
          valueCounts.plot.bar()
      else:
          columnDf.hist()
      plt.ylabel('counts')
      plt.xticks(rotation = 90)
      plt.title(f'{columnNames[i]} (column {i})')
  plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
  plt.show()

def plotCorrelationMatrix(df, graphWidth):
  # filename = df.dataframeName
  df = df.dropna('columns') # drop columns with NaN
  df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
  if df.shape[1] < 2:
      print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
      return
  corr = df.corr()
  plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
  corrMat = plt.matshow(corr, fignum = 1)
  plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
  plt.yticks(range(len(corr.columns)), corr.columns)
  plt.gca().xaxis.tick_bottom()
  plt.colorbar(corrMat)
  plt.title(f'Correlation Matrix for Food Demand Forecast', fontsize=15)
  plt.show()

def plotScatterMatrix(df, plotSize, textSize):
  df = df.select_dtypes(include =[np.number]) # keep only numerical columns
  # Remove rows and columns that would lead to df being singular
  df = df.dropna('columns')
  df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
  columnNames = list(df)
  if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
      columnNames = columnNames[:10]
  df = df[columnNames]
  ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
  corrs = df.corr().values
  for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
      ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
  plt.suptitle('Scatter and Density Plot')
  plt.show()
  

## load data from onedrive via paths set in config file
def load_data_from_one_drive(directory_to_extract_to, path_section_name, file_path_name, not_one_folder_inside=1):
  #Read config.ini file
  config_obj = configparser.ConfigParser()
  config_obj.read(str(Path(os.getcwd()).parents[not_one_folder_inside])+"\configfile.ini")
  path_to_zip_file = config_obj[path_section_name][file_path_name]
  with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
      for member in tqdm(zip_ref.infolist(), desc='Extracting '):
        zip_ref.extract(member, directory_to_extract_to)


  