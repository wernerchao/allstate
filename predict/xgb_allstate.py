# This module uses xgboost.
# We split the training data into train/test set, and further split training set
# into 3 fold CV sets.
# The predicted weights made by CV sets is used as training input for the stacking model (stacking_allstate.py).
# TODO: This may be unnecessry -> The predicted weights made by test set is used as hold-out testing input for the stacking model.

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from utilities import data_prep

if __name__ == '__main__':
    # Preprocess data for xgboost.
    train_xg = pd.read_csv('../data/train.csv')
    train_xg_x, train_xg_y = data_prep.data_prep_log(train_xg)

    test_xg = pd.read_csv('../data/test.csv') #TODO: need to preprocess the data just like the train set.
    test_xg_x, test_xg_y = data_prep.data_prep_log(test_xg, False)

    # Training xgboost on CV set and predict using out-of-fold prediction
    xgboosting = XGBRegressor(n_estimators=200, \
                            learning_rate=0.07, \
                            gamma=0.2, \
                            max_depth=8, \
                            min_child_weight=6, \
                            colsample_bytree=0.6, \
                            subsample=0.9)

    folds = KFold(n_splits=3, shuffle=False)
    for k, (train_index, test_index) in enumerate(folds.split(train_xg_x)):
        xtr = train_xg_x[train_index]
        ytr = train_xg_y[train_index]
        xtest = train_xg_x[test_index]
        ytest = train_xg_y[test_index]
        print "Fitting on fold {}...".format(k)
        print "Checking xtest shape: ", xtest.shape
        print "Checking ytest shape: ", ytest.shape
        xgboosting.fit(xtr, ytr)
        np.savetxt('xgb_pred_fold_{}.txt'.format(k), np.exp(xgboosting.predict(xtest)))
        np.savetxt('xgb_test_fold_{}.txt'.format(k), ytest)

    # Training xgboost on test set.
    xgboosting.fit(train_xg_x, train_xg_y)
    print "Fitting on test set..."
    np.savetxt('xgb_pred_test.txt', np.exp(xgboosting.predict(test_xg_x)))
