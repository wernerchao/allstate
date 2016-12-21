import tensorflow as tf
tf.python.control_flow_ops = tf

import sys

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers.advanced_activations import PReLU
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

# Preprocessing data. Note we don't log the loss.
train_mlp = pd.read_csv('train.csv')

# Making columns for features
cat_names = [i for i in train_mlp.columns if 'cat' in i]
train_mlp = pd.get_dummies(data=train_mlp, columns=cat_names)
features_mlp = [x for x in train_mlp.columns if x not in ['id','loss']]

train_mlp_x = np.array(train_mlp[features_mlp])
train_mlp_y = np.array(train_mlp['loss'])
print "train x: ", train_mlp_x[1].shape, "train y: ", train_mlp_y.shape


def mlp_cross_validation(mlp_function, train_x_data, nfolds=3):
    kf = KFold(n_splits=nfolds, shuffle=True, random_state=31337)
    val_score = np.zeros((nfolds))
    counter = 1
    for train_index, test_index in kf.split(train_x_data):
        x_train, x_test = train_mlp_x[train_index], train_mlp_x[test_index]
        y_train, y_test = train_mlp_y[train_index], train_mlp_y[test_index]
        mlp = mlp_function()
        mlp.fit(x_train, y_train, \
                validation_split=0.2, \
                batch_size=128, \
                nb_epoch=30, \
                verbose=1, \
                callbacks=[EarlyStopping(monitor="val_loss", patience=4)])
        pred = mlp.predict(x_test, batch_size=128)
        score = mean_absolute_error(pred, y_test)
        val_score[counter] = score
        print "Fold: {}, MAE Score: {}".format(counter, score)
        counter += 1
    avg_mae = sum(val_score) / nfolds
    print "{} Fold CV Score (average MAE): {}".format(nfolds, avg_mae)
    return score

# Define search space to be used in hyperopt
space = {'hidden_1_units': hp.choice('hidden_1_units', [256, 512, 768, 1024]), \
        'hidden_2_units': hp.choice('hidden_2_units', [128, 256, 512, 768]), \
        'hidden_1_dropout': hp.choice('hidden_1_dropout', [0.1, 0.6]), \
        'hidden_2_dropout': hp.choice('hidden_2_dropout', [0.1, 0.5]) \
        }


def hyperopt_search(params):
    print "Testing these parameters: ", params

    def mlp_model():
        model = Sequential()
        model.add(Dense(params['hidden_1_units'], input_dim=train_mlp_x.shape[1]))
        model.add(Activation('relu'))
        model.add(Dropout(params['hidden_1_dropout']))

        model.add(Dense(params['hidden_2_units']))
        model.add(Activation('relu'))
        model.add(Dropout(params['hidden_2_dropout']))

        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam', metrics=['mae'])
        return model

    cv_score = mlp_cross_validation(mlp_model, train_mlp_x)
    return {'loss': cv_score, 'status': STATUS_OK}

sys.stdout = open('hyperopt/hyperopt_1.log', 'w')

trials = Trials()
best = fmin(hyperopt_search, space, algo=tpe.suggest, max_evals=50, trials=trials)

