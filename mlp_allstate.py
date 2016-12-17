import tensorflow as tf
tf.python.control_flow_ops = tf

import sys

import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split

from keras.models import Sequential
from keras.models import save_model, load_model
from keras.layers.advanced_activations import PReLU
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.callbacks import EarlyStopping

# Preprocessing data. Note we don't log the loss.
train_mlp = pd.read_csv('train.csv')

# Making columns for features
cat_names = [i for i in train_mlp.columns if 'cat' in i]
train_mlp = pd.get_dummies(data=train_mlp, columns=cat_names)
features_mlp = [x for x in train_mlp.columns if x not in ['id','loss']]

train_mlp_x = np.array(train_mlp[features_mlp])
train_mlp_y = np.array(train_mlp['loss'])
print "train x: ", train_mlp_x.shape, "train y: ", train_mlp_y.shape

# mlp_x_train, mlp_x_test, mlp_y_train, mlp_y_test = train_test_split(train_mlp_x, train_mlp_y, test_size=0.25, random_state=31337)
# print "MLP Training set X: ", mlp_x_train.shape, ". Y: ", mlp_y_train.shape
# print "MLP Testing set X: ", mlp_x_test.shape, ". Y: ", mlp_y_test.shape


def mlp_model():
    model = Sequential()
    model.add(Dense(128, input_dim=train_mlp_x.shape[1]))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    return model

# Uncomment below if need to train MLP
# mlp = mlp_model()
# sys.stdout = open('mlp_v1_out.txt', 'w')
# fit = mlp.fit(train_mlp_x, train_mlp_y, validation_split=0.2, batch_size=128, nb_epoch=40, verbose=1)
# his = fit.history
# print "Validation loss by epoch 40: ", his['val_loss'][-1]



