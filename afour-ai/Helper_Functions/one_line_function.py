#imports
import pandas as pd
import pickle
import tensorflow as tf

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from sklearn.feature_selection import RFE
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import KBinsDiscretizer


# K- folds cross validation

# Shuffle the dataset randomly.
# Split the dataset into k groups
# For each unique group:
# Take the group as a hold out or test data set
# Take the remaining groups as a training data set
# Fit a model on the training set and evaluate it on the test set
# Retain the evaluation score and discard the model
# Summarize the skill of the model using the sample of model evaluation scores For more information : https://www.analyticsvidhya.com/blog/2021/06/tune-hyperparameters-with-gridsearchcv/

GridSearchCV(model, model_params, cv = cv_k)

# Example
# rfc = RandomForestRegressor()
# forest_params = [{'max_depth': list(range(10, 12)), 'max_features': list(range(0,5))}]
# clf = GridSearchCV(rfc, forest_params, cv = 3)
# clf.fit(X_train, Y_train)
# print(clf.best_params_)
# print(clf.best_score_)


# TransformedTargetRegressor transforms the targets y before fitting a regression model. The predictions are mapped back to the original space via an inverse transform. It takes as an argument the regressor that will be used for prediction, and the transformer that will be applied to the target variable:
# For more information : https://scikit-learn.org/stable/modules/compose.html#transformed-target-regressor

# returns a TransformedTargetRegressor object which can be used to fit data and get accuracy via scoring metrics eg. R2 score.
TransformedTargetRegressor(regressor=regressor, transformer=transformer, func=func, inverse_func=inverse_func, check_inverse=check_inverse)

# Function to create Keras ModelCheckPoint callback to save model and its weight at some frequency
ModelCheckpoint(f"{dirname}/model/", save_best_only=True)


# Function for learning Rate Reduction callback ReduceLROnPlateau to monitor the validation loss
# Use Case:
  # Reduce learning rate when a metric has stopped improving. 
  # Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates.
  # This callback monitors a quantity and if no improvement is seen for a 'patience' number of epochs, the learning rate is reduced
  # model.fit(X_train, Y_train, callbacks=[reduce_learning_rate(min_lr=0.01)])
# Arguments:
    # monitor: quantity to be monitored.
    # factor: factor by which the learning rate will be reduced. new_lr = lr * factor.
    # patience: number of epochs with no improvement after which learning rate will be reduced.
    # verbose: int. 0: quiet, 1: update messages.
    # mode: one of {'auto', 'min', 'max'}. In 'min' mode, the learning rate will be reduced when the quantity monitored has stopped decreasing; in 'max' mode it will be reduced when the quantity monitored has stopped increasing; in 'auto' mode, the direction is automatically inferred from the name of the monitored quantity.
    # min_delta: threshold for measuring the new optimum, to only focus on significant changes.
    # cooldown: number of epochs to wait before resuming normal operation after lr has been reduced. min_lr: lower bound on the learning rate.
ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=min_lr)


# Function for finding the ideal learning rate using LearningRateSchedular callback
# Use Case:
  # Scheduler is a function which takes 2 arguments (epoch and LR) and updates the learning rate.
  # This scheduler function is then called by the LearningRateScheduler callback
  # LearningRateScheduler callback gets a new learning rate at the beginning of every epoch from
  # a scheduler function and applies this updated LR to the optimizer
  # model.fit(X_train, Y_train, epochs=10, callbacks=[learning_rate_schedular(min_lr=0.01)])
def sample_scheduler(epoch, lr):
  """
  This schedular function keeps the initial learning rate for the first ten epochs
  and decreases it exponentially after that.

  Args:
    epoch: epoch value
    lr: learning rate
  """
  if epoch < 10:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

tf.keras.callbacks.LearningRateScheduler(sample_scheduler)


# Function for data Augmentation callback for a given image using keras ImageDataGenerator
# Use Case:
  # ImageDataGenerator class allows you to randomly rotate images through any degree
  # between 0 and 360 by providing an integer value in the rotation_range argument.
tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False, samplewise_center=False,
    featurewise_std_normalization=False, samplewise_std_normalization=False,
    zca_whitening=False, zca_epsilon=1e-06, rotation_range=rotation_range, width_shift_range=width_shift_range,
    height_shift_range=height_shift_range, brightness_range=brightness_range, shear_range=0.0, zoom_range=0.0,
    channel_shift_range=0.0, fill_mode='nearest', cval=0.0,
    horizontal_flip=False, vertical_flip=False, rescale=None,
    preprocessing_function=None, data_format=None, validation_split=0.0, dtype=None
)


# Function for Unzipping files
!unzip "path"

# Optimal number of epochs using EarlyStopping callback
  """
  Usage:
  history = model.fit(X_train, y_train, batch_size = 128, 
                      epochs = 25, validation_data =(X_valid, y_valid), 
                      callbacks = earlystop())
  """

EarlyStopping(monitor ="val_loss", mode ="min", patience = 5, restore_best_weights = True)


# Piyush check and change the parameters
# Saving a model at loc = file_loc
pickle.dump(model, open(file_loc, 'wb'))

#Loading the saved model from the loc = file_loc
pickle.load(open(file_loc, 'rb'))

#ColumnTransfer
#it appies various data cleaning functions for the given columns
col_transform = ColumnTransformer([('num',MinMaxScaler, [0,1]), ('cat', OneHotEncoder, slice(2,4))]).fit_transform(X)

#Label Encoder
df_new = LabelEncoder().fit_transform(X)

#OneHot Encoder
df_new = pd.get_dummies(data=df, drop_first=True, columns = [[cat_col]])

#EarlyStopping callback
earlystopping = EarlyStopping(monitor ="val_loss", mode ="min", patience = 5, restore_best_weights = True)

#Finding the optimum value of p, d and q in an ARIMA model using autoARIMA
model_autoARIMA = auto_arima(train, start_p=0, start_q=0,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=5, max_q=5, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
print(model_autoARIMA.summary())

#Recursive Feature Elimination(RFE)
"""
estimator : Estimator instance
A supervised learning estimator with a fit method that provides information about feature importance (e.g. coef_, feature_importances_).

n_features_to_select : int or float, default=None
The number of features to select. If None, half of the features are selected. If integer, the parameter is the absolute number of features to select. If float between 0 and 1, it is the fraction of features to select.

step : int or float, default=1
If greater than or equal to 1, then step corresponds to the (integer) number of features to remove at each iteration. If within (0.0, 1.0), then step corresponds to the percentage (rounded down) of features to remove at each iteration.
"""
selector = RFE(estimator, n_features_to_select=5, step=1)
selector = selector.fit(X, y)


# Detect outliers with IsolationForest
"""
+1 -> inlier
-1 -> outliner

Let X = [[[-1.1], [0.3], [0.5], [100]]]
"""
check_outlier = IsolationForest(random_state=0).fit(X)
# check_outlier.predict([[0.1], [0], [90]])
# Returns array([ 1,  1, -1])

# Detect outliers with LocalOutlierFactor
"""
n_neighbors : int, default=20
Number of neighbors to use by default for kneighbors queries. If n_neighbors is larger than the number of samples provided, all samples will be used.

X = [[-1.1], [0.2], [101.1], [0.3]]
"""
clf = LocalOutlierFactor(n_neighbors=2)
clf.fit_predict(X)
#Returns array([ 1,  1, -1,  1])

# K-bins-Descritization to convert numeric data into binary or into multiple categories based on the threshold
"""
n_bins : int or array-like of shape (n_features,), default=5
The number of bins to produce. Raises ValueError if n_bins < 2.

encode : {‘onehot’, ‘onehot-dense’, ‘ordinal’}, default=’onehot’
Method used to encode the transformed result.

‘onehot’: Encode the transformed result with one-hot encoding and return a sparse matrix. Ignored features are always stacked to the right.

‘onehot-dense’: Encode the transformed result with one-hot encoding and return a dense array. Ignored features are always stacked to the right.

‘ordinal’: Return the bin identifier encoded as an integer value.

strategy : {‘uniform’, ‘quantile’, ‘kmeans’}, default=’quantile’
Strategy used to define the widths of the bins.

‘uniform’: All bins in each feature have identical widths.

‘quantile’: All bins in each feature have the same number of points.

‘kmeans’: Values in each bin have the same nearest center of a 1D k-means cluster.
X = [[-2, 1, -4,   -1],
     [-1, 2, -3, -0.5],
     [ 0, 3, -2,  0.5],
     [ 1, 4, -1,    2]]
"""
est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform').fit(X)
# Xt = est.transform(X)
# array([[ 0., 0., 0., 0.],
#        [ 1., 1., 1., 0.],
#        [ 2., 2., 2., 1.],
#        [ 2., 2., 2., 2.]])


