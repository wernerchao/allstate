import numpy as np
import pandas as pd

from sklearn.cross_validation import KFold, train_test_split

import xgboost as xgb

# Dealing with xgboost data
train_xg = pd.read_csv('train.csv')
n_train = len(train_xg)
train_xg['log_loss'] = np.log(train_xg['loss'])

# Creating columns for features, and categorical features
features_col = [x for x in train_xg.columns if x not in ['id', 'loss', 'log_loss']]
cat_features_col = [x for x in train_xg.select_dtypes(include=['object']).columns if x not in ['id', 'loss', 'log_loss']]

train_xg_x = np.array(train_xg[features_col])
train_xg_y = np.array(train_xg['log_loss'])

xg_x_train, xg_x_test, xg_y_train, xg_y_test  = train_test_split(train_xg_x, train_xg_y, test_size=0.25, random_state=31337)
print "XGB Training set X: ", xg_x_train.shape, ". Y: ", xg_y_train.shape
print "XGB Testing set X: ", xg_x_test.shape, ". Y: ", xg_y_test.shape


# Dealing with MLP data. Note we don't log the loss.
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

