import numpy as np
import pandas as pd

from sklearn.cross_validation import KFold, train_test_split

import xgboost as xgb
from xgboost import XGBRegressor


# Dealing with xgboost data
train_xg = pd.read_csv('train.csv')
n_train = len(train_xg)
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

# Training xgboost
folds = KFold(len(xg_x_train), shuffle=False, n_folds=3)
for k, (train_index, test_index) in enumerate(folds):
    xtr = xg_x_train[train_index]
    ytr = xg_y_train[train_index]
    xtest = xg_x_train[test_index]
    ytest = xg_y_train[test_index]
    xgboosting = XGBRegressor(n_estimators=200, \
                           learning_rate=0.07, \
                           gamma=0.2, \
                           max_depth=8, \
                           min_child_weight=6, \
                           colsample_bytree=0.6, \
                           subsample=0.9)
    xgboosting.fit(xtr, ytr)
    # np.savetxt('ensemble/xgb_pred_fold_{}.txt'.format(k), np.exp(xgboosting.predict(xtest)))
    # np.savetxt('ensemble/xgb_test_fold_{}.txt'.format(k), ytest)







