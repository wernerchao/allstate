import numpy as np
import pandas as pd

from utilities import data_prep

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error


def load_model(model_name, n_cv=3):
    ''' Input the model name to be loaded, and n_folds used.
    Returns the model that is aggregated from weights predicted from CV sets. '''

    train = []
    for i in xrange(n_cv):
        train.append(np.loadtxt('model_output/{}_pred_fold_{}.txt'.format(model_name, i)))
    return train


if __name__ == '__main__':
    # (1) Preprocessing data for MLP. Note that we don't log the output variable.
    train_mlp = pd.read_csv('../data/train.csv')
    train_mlp_x, train_mlp_y = data_prep.data_prep(train_mlp, True)
    mlp_x_train, mlp_x_test, mlp_y_train, mlp_y_test = train_test_split(train_mlp_x, train_mlp_y, test_size=0.25, random_state=31337)


    # (2) Aggregate weights to be passed into layer 2. Train set for layer 2 model.
    # (2.1) This is xgb predicted weights from Kfold training set
    train_xgb = load_model('xgb', 3)
    train_xgb_folds = np.hstack((train_xgb[0], train_xgb[1], train_xgb[2]))
    print "Checking XGB training set size: ", train_xgb_folds.shape

    mae_check_1 = mean_absolute_error(mlp_y_train, train_xgb_folds)
    print "XGBoost prediction error: ", mae_check_1

    # (2.2) This is MLP predicted weights from Kfold training set
    train_mlp = load_model('mlp', 3)
    train_mlp_folds = np.hstack((train_mlp[0], train_mlp[1], train_mlp[2]))
    print "Checking MLP trainin set size: ", train_mlp_folds.shape

    mae_check_2 = mean_absolute_error(mlp_y_train, train_mlp_folds)
    print "MLP prediction error: ", mae_check_2


    # (3) Load predicted weights made on whole training set. CV set for layer 2 model.
    print 'Loading weghts from models (prediction made on whole data set) as test sets...'
    cv_xgb = np.loadtxt('model_output/xgb_pred_test.txt')
    cv_mlp = np.loadtxt('model_output/mlp_pred_test.txt')
    print "cv_mlp: ", cv_mlp.shape, "cv_xgb: ", cv_xgb.shape


    # (4) Load predicted weights made on test set. Test set for layer 2 model.
    print 'Loading weghts from models (prediction made on whole data set) as test sets...'
    test_xgb = np.loadtxt('model_output_2/xgb_pred_test.txt')
    test_mlp = np.loadtxt('model_output_2/mlp_pred_test.txt')
    print "test_mlp: ", test_mlp.shape, "test_xgb: ", test_xgb.shape


    # (5) Train layer 2 model. We use linear regression
    print 'Aggregating XGBoost and MLP weights as x_train & x_test...'
    layer_1_train_x = np.vstack((train_xgb_folds, train_mlp_folds)).T
    layer_1_cv_x = np.vstack((cv_xgb, cv_mlp)).T
    layer_1_test_x = np.vstack((test_xgb, test_mlp)).T
    print "layer_1_train_x: ", layer_1_train_x.shape
    print "layer_1_cv_x: ", layer_1_cv_x.shape
    print "layer_1_test_x: ", layer_1_test_x.shape

    layer_1_train_y = mlp_y_train # Note: it's not logged
    layer_1_cv_y = mlp_y_test # Note: it's not logged
    print "layer_1_train_y, ", layer_1_train_y.shape
    print "layer_1_cv_y, ", layer_1_cv_y.shape

    print 'Stacking with linear regression...'
    reg = LinearRegression()
    reg.fit(np.log(layer_1_train_x), np.log(layer_1_train_y))
    pred = reg.predict(np.log(layer_1_cv_x))
    holdout_mae = mean_absolute_error(np.exp(pred), layer_1_cv_y)
    print "Stacker MAE (hold out): ", holdout_mae

    # (6) Output final prediction and save to csv file.
    test_set = pd.read_csv('../data/test.csv')
    final_pred = reg.predict(layer_1_test_x)
    print "Outputing final prediction......\n shape is:", final_pred.shape
    final_df = pd.DataFrame({'id':test_set.id, 'loss':final_pred})
    final_df.to_csv('submission_1.csv', index=False)
