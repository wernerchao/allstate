import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold, train_test_split
from sklearn.metrics import mean_absolute_error

import xgboost as xgb
from xgboost import XGBRegressor


# Dealing with xgboost data
train_xg = pd.read_csv('train.csv')
n_train = len(train_xg)
print "Whole data set size: ", n_train
train_xg['log_loss'] = np.log(train_xg['loss'])

# Creating columns for features, and categorical features
features_col = [x for x in train_xg.columns if x not in ['id', 'loss', 'log_loss']]
cat_features_col = [x for x in train_xg.select_dtypes(include=['object']).columns if x not in ['id', 'loss', 'log_loss']]
for c in range(len(cat_features_col)):
    train_xg[cat_features_col[c]] = train_xg[cat_features_col[c]].astype('category').cat.codes

train_xg_x = np.array(train_xg[features_col])
train_xg_y = np.array(train_xg['log_loss'])

xg_x_train, xg_x_test, xg_y_train, xg_y_test  = train_test_split(train_xg_x, train_xg_y, test_size=0.25, random_state=31337)
print "XGB Training set X: ", xg_x_train.shape, ". Y: ", xg_y_train.shape
print "XGB Testing set X: ", xg_x_test.shape, ". Y: ", xg_y_test.shape


# Dealing with MLP data. Note: we don't log the loss.
train_mlp = pd.read_csv('train.csv')

# Making columns for features
cat_names = [i for i in train_mlp.columns if 'cat' in i]
train_mlp = pd.get_dummies(data=train_mlp, columns=cat_names)
features_mlp = [x for x in train_mlp.columns if x not in ['id','loss']]

train_mlp_x = np.array(train_mlp[features_mlp])
train_mlp_y = np.array(train_mlp['loss'])

mlp_x_train, mlp_x_test, mlp_y_train, mlp_y_test = train_test_split(train_mlp_x, train_mlp_y, test_size=0.25, random_state=31337)
print "MLP Training set X: ", mlp_x_train.shape, ". Y: ", mlp_y_train.shape
print "MLP Testing set X: ", mlp_x_test.shape, ". Y: ", mlp_y_test.shape



# ### Layer 1 Model. Xgboost only. MLP has been trained separately.
# # Training xgboost. Out-of-fold prediction
# folds = KFold(len(xg_x_train), shuffle=False, n_folds=3)
# for k, (train_index, test_index) in enumerate(folds):
#     xtr = xg_x_train[train_index]
#     ytr = xg_y_train[train_index]
#     xtest = xg_x_train[test_index]
#     ytest = xg_y_train[test_index]
#     print "xtest shape: ", xtest.shape
#     print "ytest shape: ", ytest.shape
#     xgboosting = XGBRegressor(n_estimators=200, \
#                            learning_rate=0.07, \
#                            gamma=0.2, \
#                            max_depth=8, \
#                            min_child_weight=6, \
#                            colsample_bytree=0.6, \
#                            subsample=0.9)
#     xgboosting.fit(xtr, ytr)
#     # np.savetxt('ensemble/xgb_pred_fold_{}.txt'.format(k), np.exp(xgboosting.predict(xtest)))
#     # np.savetxt('ensemble/xgb_test_fold_{}.txt'.format(k), ytest)


# # Training xgboost on the whole set.
# xgboosting = XGBRegressor(n_estimators=200, \
#                            learning_rate=0.07, \
#                            gamma=0.2, \
#                            max_depth=8, \
#                            min_child_weight=6, \
#                            colsample_bytree=0.6, \
#                            subsample=0.9)
# xgboosting.fit(xg_x_train, xg_y_train)
# # np.savetxt('ensemble/xgb_pred_test.txt'.format(k), np.exp(xgboosting.predict(xg_x_test)))


### Aggregate weights to be passed into layer 2 model
# This is y(train_xg['log_loss']) label from Kfold validation set
test_xgb_1 = np.exp(np.loadtxt('ensemble/xgb_test_fold_0.txt'))
test_xgb_2 = np.exp(np.loadtxt('ensemble/xgb_test_fold_1.txt'))
test_xgb_3 = np.exp(np.loadtxt('ensemble/xgb_test_fold_2.txt'))
print "Checking validation set size: ", test_xgb_1.shape
test_xgb_folds = np.hstack((test_xgb_1, test_xgb_2, test_xgb_3))

mlp_test_fold_1 = np.loadtxt('ensemble/mlp_test_fold_0.txt')
mlp_test_fold_2 = np.loadtxt('ensemble/mlp_test_fold_1.txt')
mlp_test_fold_3 = np.loadtxt('ensemble/mlp_test_fold_2.txt')
test_mlp_folds = np.hstack((mlp_test_fold_1, mlp_test_fold_2, mlp_test_fold_3))

mae_check_1 = mean_absolute_error(test_mlp_folds, test_xgb_folds) # This should be 0
print mae_check_1

# This is xgb predicted weights from Kfold training set
train_xgb_1 = np.loadtxt('ensemble/xgb_pred_fold_0.txt')
train_xgb_2 = np.loadtxt('ensemble/xgb_pred_fold_1.txt')
train_xgb_3 = np.loadtxt('ensemble/xgb_pred_fold_2.txt')
train_xgb_folds = np.hstack((train_xgb_1, train_xgb_2, train_xgb_3))
print "Checking training set size: ", train_xgb_folds.shape

mae_check_2 = mean_absolute_error(np.exp(xg_y_train), train_xgb_folds)
print "XGBoost prediction error: ", mae_check_2

# This is MLP predicted weights from Kfold training set
train_mlp_1 = np.loadtxt('ensemble/mlp_pred_fold_0.txt')
train_mlp_2 = np.loadtxt('ensemble/mlp_pred_fold_1.txt')
train_mlp_3 = np.loadtxt('ensemble/mlp_pred_fold_2.txt')
train_mlp_folds = np.hstack((train_mlp_1, train_mlp_2, train_mlp_3))
print "Checking trainin set size again: ", train_mlp_folds.shape

mae_check_3 = mean_absolute_error(mlp_y_train, train_mlp_folds)
print "MLP prediction error: ", mae_check_3


# Load the weights from training the whole data set as test sets
test_xgb = np.loadtxt('ensemble/xgb_pred_test.txt')
test_mlp = np.loadtxt('ensemble/mlp_pred_test.txt')

### Train layer 2 model. We use linear regression
# Aggregate the XGBoost and MLP weights as X
layer_1_train_x = np.vstack((train_xgb_folds, train_mlp_folds)).T
layer_1_test_x = np.vstack((test_xgb, test_mlp)).T
layer_1_train_y = mlp_y_train
layer_1_test_y = mlp_y_test
print "Xtrain shape:", layer_1_train_x.shape
print "ytrain shape:", layer_1_train_y.shape
print "Xtest shape:", layer_1_test_x.shape
print "ytest shape:", layer_1_test_y.shape


reg = LinearRegression()
reg.fit(np.log(layer_1_train_x), np.log(layer_1_train_y))
pred = reg.predict(np.log(layer_1_test_x))
final_score = reg.score(np.log(layer_1_test_x), layer_1_test_y)
final_mae = mean_absolute_error(layer_1_test_y, np.exp(pred))

print "Final stacker MAE: ", final_mae
print "Final Score: ", final_score










