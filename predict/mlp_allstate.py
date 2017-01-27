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


if __name__ == '__main__':
    # Preprocessing data. Note that we don't log the output variable.
    train_mlp = pd.read_csv('../data/train.csv')
    train_mlp_x, train_mlp_y = data_prep.data_prep(train_mlp, True)

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
    out_file = open('mlp_v1_out_1.txt', 'w')
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
