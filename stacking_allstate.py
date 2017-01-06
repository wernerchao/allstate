# This module stacks xgboost and MLP prediction using LinearRegression.
# Predicted weights from CV sets is used as training features.
# Predicted weights from test set is used as testing features.
# Final Stacker MAE: 1136.0091763. 
# Compared to single model xgboost: 1151.51901659, MLP: 1146.52636433.

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold, train_test_split
from sklearn.metrics import mean_absolute_error


# Preprocessing data for MLP. Note that we don't log the output variable.
train_mlp = pd.read_csv('train.csv')

# Making columns for features
cat_names = [i for i in train_mlp.columns if 'cat' in i]
train_mlp = pd.get_dummies(data=train_mlp, columns=cat_names)
features_mlp = [x for x in train_mlp.columns if x not in ['id','loss']]

train_mlp_x = np.array(train_mlp[features_mlp])
train_mlp_y = np.array(train_mlp['loss'])
print "train x: ", train_mlp_x[1].shape, "train y: ", train_mlp_y.shape

mlp_x_train, mlp_x_test, mlp_y_train, mlp_y_test = train_test_split(train_mlp_x, train_mlp_y, test_size=0.25, random_state=31337)


### Aggregate weights to be passed into layer 2 model
# This is xgb predicted weights from Kfold training set
train_xgb_1 = np.loadtxt('ensemble/xgb_pred_fold_0.txt')
train_xgb_2 = np.loadtxt('ensemble/xgb_pred_fold_1.txt')
train_xgb_3 = np.loadtxt('ensemble/xgb_pred_fold_2.txt')
train_xgb_folds = np.hstack((train_xgb_1, train_xgb_2, train_xgb_3))
print "Checking XGB training set size: ", train_xgb_folds.shape

mae_check_1 = mean_absolute_error(mlp_y_train, train_xgb_folds)
print "XGBoost prediction error: ", mae_check_1

# This is MLP predicted weights from Kfold training set
train_mlp_1 = np.loadtxt('ensemble/mlp_pred_fold_0.txt')
train_mlp_2 = np.loadtxt('ensemble/mlp_pred_fold_1.txt')
train_mlp_3 = np.loadtxt('ensemble/mlp_pred_fold_2.txt')
train_mlp_folds = np.hstack((train_mlp_1, train_mlp_2, train_mlp_3))
print "Checking MLP trainin set size: ", train_mlp_folds.shape

mae_check_2 = mean_absolute_error(mlp_y_train, train_mlp_folds)
print "MLP prediction error: ", mae_check_2


# Load the weights from training the whole data set as test sets
test_xgb = np.loadtxt('ensemble/xgb_pred_test.txt')
test_mlp = np.loadtxt('ensemble/mlp_pred_test.txt')

### Train layer 2 model. We use linear regression
# Aggregate the XGBoost and MLP weights as X
layer_1_train_x = np.vstack((train_xgb_folds, train_mlp_folds)).T

layer_1_test_x = np.vstack((test_xgb, test_mlp)).T
print test_xgb.shape

layer_1_train_y = mlp_y_train
layer_1_test_y = mlp_y_test
print "Xtrain shape:", layer_1_train_x.shape
print "ytrain shape:", layer_1_train_y.shape
print "Xtest shape:", layer_1_test_x.shape
print "ytest shape:", layer_1_test_y.shape


reg = LinearRegression()
reg.fit(np.log(layer_1_train_x), np.log(layer_1_train_y))
pred = reg.predict(np.log(layer_1_test_x))
final_mae = mean_absolute_error(pred, layer_1_test_y) #TODO: need to log or exp
print "Final stacker MAE: ", final_mae
