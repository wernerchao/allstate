# This module runs a quick 3-layer neural network score check. We use MLP in for this problem.
# We can quickly know how well MLP fits the data.
# We also implement a more complex 4-layer MLP to look for direction for tuning the hyperparameters.

import tensorflow as tf

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error

from keras.models import Sequential
from keras.models import save_model, load_model
from keras.layers.advanced_activations import PReLU
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.callbacks import EarlyStopping

from utilities import data_prep


def mlp_model():
    ''' A 4-layer MLP model. Input layer 1153 nodes, hidden layer 256 nodes, 
        and output layer 1 node. '''

    model = Sequential()
    model.add(Dense(351, input_dim=train_mlp_x.shape[1], init='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.578947))

    model.add(Dense(293, init='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.26666))

    model.add(Dense(46, init='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.188888))

    model.add(Dense(1, init='glorot_normal'))
    model.compile(loss='mae', optimizer='adadelta')
    return model


def cross_validate_mlp(train_x, train_y, mlp_func, nfolds=3):
    folds = KFold(n_splits=nfolds, shuffle=True, random_state=31337)
    val_scores = np.zeros((nfolds,))
    for k,(train_index, test_index) in enumerate(folds):
        mlp = mlp_func()
        xtr, ytr = train_x[train_index], train_y[train_index]
        xte, yte = train_x[test_index], train_y[test_index]
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        fit = mlp.fit(xtr, ytr, batch_size=128, \
                      nb_epoch=30, verbose=1, callbacks=[early_stopping])
        pred = mlp.predict(xte, batch_size=256)
        score = mean_absolute_error(yte, pred)
        val_scores[k] += score
        print 'Fold {}, MAE: {}'.format(k, score)
        np.savetxt('ensemble/mlp_pred_fold_{}.txt'.format(k), pred)
        np.savetxt('ensemble/mlp_test_fold_{}.txt'.format(k), yte)
    avg_score = np.sum(val_scores) / float(nfolds)
    print '{}-fold CV score: {}'.format(nfolds, avg_score)
    return avg_score


def plot_mlp(hist, title):
    ''' A histogram of training loss & validation loss'''

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title(title)
    ax1.plot(hist['loss'], label='Training Loss')
    ax1.plot(hist['val_loss'], label='Validation Loss')
    ax1.legend()

    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax1.set_title('Last 20 Epochs, {}'.format(title))
    ax2.plot(hist['loss'][20:40], label='Training Loss')
    ax1.plot(hist['val_loss'][20:40], label='Validation Loss')
    ax2.legend()
    plt.show()


if __name__ == '__main__':
    ''' Step (1) Preprocess the data_prep.
    Step (2) Train the model and plot the overfit.
    Step (3) Train & predcit the model with 3 fold CV  train data. Predicted weights are for ensemble train set.
    Step (4) Train & Predict the model with train/test data. Predicted weights are for ensemble test set. '''

    # (1) Preprocessing data. Note that we don't log the output variable.
    train_mlp = pd.read_csv('../data/train.csv')
    train_mlp_x, train_mlp_y = data_prep.data_prep(train_mlp, True)
    test_mlp = pd.read_csv('../data/test.csv')
    test_mlp_x, test_mlp_y = data_prep.data_prep(test_mlp, False)

    # (2) Train MLP & output file.
    mlp_overfit = mlp_model()
    saveout = sys.stdout
    out_file = open('mlp_overfit_out.txt', 'w')
    sys.stdout = out_file
    fit = mlp_overfit.fit(train_mlp_x, train_mlp_y, validation_split=0.2, batch_size=128, nb_epoch=40, verbose=1)
    hist = fit.history
    print "Validation loss by epoch 40: ", hist['val_loss'][-1]
    print "History has: ", hist
    sys.stdout = saveout
    out_file.close()

    # Plot the fitting history, shows overfitting
    models_history = {}
    models_history['mlp_1'] = hist
    plot_mlp(models_history['mlp_1'], 'mlp_1')

    # (3) Train/predict with 3 fold CV train data. Save predicted weights as ensemble train set.
    mlp_final = mlp_model()
    cv_score = cross_validate_mlp(train_mlp_x, train_mlp_y, mlp_final, 3)
    print cv_score

    # (4) Train/predict with train/test data. Save predicted weights as ensemble test set.
    mlp_final.fit(train_mlp_x, train_mlp_y, batch_size=128, nb_epoch=30, verbose=1)
    pred_mlp = mlp_final.predict(test_mlp_x, batch_size=256)
    np.savetxt('mlp_pred_test.txt', pred_mlp)
