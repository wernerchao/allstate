# Allstate Claims Severity
This repository is for building a model for the Kaggle challenge: [Allstate Claims Severity](https://www.kaggle.com/c/allstate-claims-severity).
A regression problem for predicting insurance severity.

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