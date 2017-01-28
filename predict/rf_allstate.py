# This module uses RandomForestRegressor to do a baseline score check.
# RF MAE: 1246.819912
# It also tune the RF model using hyperopt.

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score #TODO: Delete.
from sklearn.ensemble import RandomForestRegressor
from utilities import data_prep


if __name__ == '__main__':
    # Preprocessing data
    train_rf = pd.read_csv('../data/train.csv')
    train_rf_x, train_rf_y = data_prep.data_prep_log(train_rf, False)
    train_rf_y = train_rf['loss']
    rf_x_train, rf_x_test, rf_y_train, rf_y_test  = train_test_split(train_rf_x, train_rf_y, test_size=0.25, random_state=31337)

    # Quick check of RF score below.
    print 'Fitting randomforest......'
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, min_samples_split=4, n_jobs=2)
    rf.fit(rf_x_train, rf_y_train)
    pred = rf.predict(rf_x_test)
    mae = mean_absolute_error(pred, rf_y_test)
    score = rf.score(rf_x_test, rf_y_test)
    print "Random Forest MAE: ", mae
    print "Random Forest Score: ", score
