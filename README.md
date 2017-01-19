# Allstate Claims Severity
This repository is for building a model for the Kaggle challenge: [Allstate Claims Severity](https://www.kaggle.com/c/allstate-claims-severity).
A regression problem for predicting insurance severity.

# This module stacks xgboost and MLP prediction using LinearRegression.
# Predicted weights from CV sets is used as training features.
# Predicted weights from test set is used as testing features.
# Final Stacker MAE: 1136.0091763. 
# Compared to single model xgboost: 1151.51901659, MLP: 1146.52636433.

Overall I achieved MAE of 1136.009 using a stacking method of xgboost and multi-layer perceptron.

The repository is organized as follows:
1. Data Exploratory - explore_allstate.py
2. Random Forest Baseline Model - rf_allstate.py
3. XGBoost - xgb_allstate.py
4. Neural Network Version 1 - mlp_allstate.py
5. Neural Network with Hyperopt - mlp_hyperopt.py
6. Stacked Model - stacking_allstate.py

The models & ensemble folders consist of the output from xgboost and MLP models, 
which are then stacked by the linear model in the stacking.allstate.py.

######################################################
Folder Structure:
data - train.csv
     - test.csv

explore - explore_allstate.py: data exploratory, generate graphs and explore data

predict - rf_allstate.py: generate prediction file with random forest
        - xgb_allstate.py: ...... with xgboost
        - mlp_allstate.py: ...... with MLP
        - stacking_allstate: stack all predictors together
        utilities - data_preprocess.py: preprocess the data, fill in missing values, etc...
        model_output - xgb_pred_fold_0.txt # predicted weights from xgb with CV set -> train set for stacking
                     - xgb_pred_fold_1.txt
                     - xgb_pred_fold_2.txt
                     - xgb_pred_test.txt # predicted wegihts from xgb with whole train set -> test set for stacking




