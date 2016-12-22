import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error
# from sklearn.cross_validation import train_test_split

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.mongoexp import MongoTrials

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

# Making custom CV sets. Uncomment below to check score for unoptimized RF
# xg_x_train, xg_x_test, xg_y_train, xg_y_test  = train_test_split(train_xg_x, train_xg_y, test_size=0.25, random_state=31337)
# print "RF Training set X: ", xg_x_train.shape, ". Y: ", xg_y_train.shape
# print "RF Testing set X: ", xg_x_test.shape, ". Y: ", xg_y_test.shape


space_rf = {
    # 'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2']),
    # 'n_estimators': hp.choice('n_estimators', [100, 500, 1000]),
    'min_samples_leaf': hp.choice('min_samples_leaf', range(1,100))
}

trials = MongoTrials('mongo://localhost:1234/foo_db/jobs', exp_key='exp1')


def minMe(params):
    # Hyperopt tuning for hyperparameters
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestRegressor
    from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

    try:
        import dill as pickle
        print('Went with dill')
    except ImportError:
        import pickle

    def hyperopt_rf(params):
        rf = RandomForestRegressor(**params)
        return cross_val_score(rf, train_xg_x, train_xg_y).mean()

    acc = hyperopt_rf(params)
    print 'new acc:', acc, 'params: ', params
    return {'loss': -acc, 'status': STATUS_OK}

best = fmin(fn=minMe, space=space_rf, trials=trials, algo=tpe.suggest, max_evals=100)


print "Best: ", best

# Uncomment below to see unoptimized RF score
# rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, min_samples_split=4)
# rf.fit(xg_x_train, xg_y_train)
# pred = rf.predict(xg_x_test)
# mae = mean_absolute_error(pred, xg_y_test)
# score = rf.score(xg_x_test, xg_y_test)
# print "Random Forest MAE: ", mae
# print "Random Forest Score: ", score


