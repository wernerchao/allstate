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

# Preprocessing data. Note that we don't log the output variable.
train_mlp = pd.read_csv('train.csv')

# Making columns for features
cat_names = [i for i in train_mlp.columns if 'cat' in i]
train_mlp = pd.get_dummies(data=train_mlp, columns=cat_names)
features_mlp = [x for x in train_mlp.columns if x not in ['id','loss']]

train_mlp_x = np.array(train_mlp[features_mlp])
train_mlp_y = np.array(train_mlp['loss'])
print "train x: ", train_mlp_x[1].shape, "train y: ", train_mlp_y.shape

mlp_x_train, mlp_x_test, mlp_y_train, mlp_y_test = train_test_split(train_mlp_x, train_mlp_y, test_size=0.25, random_state=31337)
print "MLP Training set X: ", mlp_x_train.shape, ". Y: ", mlp_y_train.shape
print "MLP Testing set X: ", mlp_x_test.shape, ". Y: ", mlp_y_test.shape


# A quick MLP model check to see how it performs. Model version 1.
# 3 layer model: 1 input, 1 hidden, 1 output layer.
def mlp_model_1():
    ''' A 4-layer MLP model. Input layer 1153 nodes, hidden layer 256 nodes, 
        and output layer 1 node. '''

    model = Sequential()
    model.add(Dense(256, input_dim=train_mlp_x.shape[1]))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    return model

# Train MLP & output file.
mlp = mlp_model_1()
saveout = sys.stdout
out_file = open('models/mlp_v1_out_1.txt', 'w')
sys.stdout = out_file
fit = mlp.fit(train_mlp_x, train_mlp_y, validation_split=0.2, batch_size=128, nb_epoch=40, verbose=1)
hist = fit.history
print "Validation loss by epoch 40: ", hist['val_loss'][-1]
print "History has: ", hist
sys.stdout = saveout
out_file.close()

# Make a histogram plot of the output.
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

models_history = {}
models_history['mlp_1'] = hist
plot_mlp(models_history['mlp_1'], 'mlp_1')




# Now that we know how the simple MLP performs. We will add more layers for a more complex model.
# A 4-layer model: 1 input, 2 hidden, 1 output layer.
def mlp_model_2():
    ''' A 4-layer MLP model. Input layer 1153 nodes, 1st hidden layer 128 nodes,
        2nd layer 64 nodes, and output layer 1 node. '''

    model = Sequential()
    model.add(Dense(128, input_dim=train_mlp_x.shape[1]))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    return model


# As we are building a more complex model, it might overfit. We setup a CV check to make sure we don't overfit.
def mlp_cross_validation(mlp_function, train_x_data, nfolds=3):
    ''' A custom CV check to prevent overfitting '''
    kf = KFold(n_splits=nfolds, shuffle=True, random_state=31337)
    val_score = np.zeros((nfolds))

    for counter, (train_index, test_index) in enumerate(kf.split(train_x_data)):
        x_train, x_test = train_mlp_x[train_index], train_mlp_x[test_index]
        y_train, y_test = train_mlp_y[train_index], train_mlp_y[test_index]
        mlp = mlp_function()

        # Fit and output log file.
        saveout = sys.stdout
        out_file = open('models/mlp_cv_v2_out_{}.txt'.format(counter), 'w')
        sys.stdout = out_file
        fit = mlp.fit(x_train, y_train, \
                      validation_split=0.2, \
                      batch_size=128, \
                      nb_epoch=30, \
                      verbose=1, \
                      callbacks=[EarlyStopping(monitor="val_loss", patience=4)])
        hist = fit.history
        print "Validation loss by epoch 30: ", hist['val_loss'][-1]
        print "History has: ", hist

        # Predict.
        pred = mlp.predict(x_test, batch_size=128)
        score = mean_absolute_error(pred, y_test)
        val_score[counter] = score
        print "Fold: {}, MAE Score: {}".format(counter, score)
        sys.stdout = saveout
        out_file.close()

    avg_mae = sum(val_score) / nfolds
    print "{} Fold CV Score (average MAE): {}".format(nfolds, avg_mae)
    return avg_mae

mlp_cross_validation(mlp_model_2, train_mlp_x, 3)
