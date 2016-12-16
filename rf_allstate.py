import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import train_test_split


# Preprocessing data
train_xg = pd.read_csv('train.csv')
n_train = len(train_xg)
print "Whole data set size: ", n_train

# Creating columns for features, and categorical features
features_col = [x for x in train_xg.columns if x not in ['id', 'loss', 'log_loss']]
cat_features_col = [x for x in train_xg.select_dtypes(include=['object']).columns if x not in ['id', 'loss', 'log_loss']]
for c in range(len(cat_features_col)):
    train_xg[cat_features_col[c]] = train_xg[cat_features_col[c]].astype('category').cat.codes

train_xg_x = np.array(train_xg[features_col])
train_xg_y = np.array(train_xg['loss'])

xg_x_train, xg_x_test, xg_y_train, xg_y_test  = train_test_split(train_xg_x, train_xg_y, test_size=0.25, random_state=31337)
print "RF Training set X: ", xg_x_train.shape, ". Y: ", xg_y_train.shape
print "RF Testing set X: ", xg_x_test.shape, ". Y: ", xg_y_test.shape

rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, min_samples_split=4)
rf.fit(xg_x_train, xg_y_train)
pred = rf.predict(xg_x_test)
mae = mean_absolute_error(pred, xg_y_test)
score = rf.score(xg_x_test, xg_y_test)
print "Random Forest MAE: ", mae
print "Random Forest Score: ", score


