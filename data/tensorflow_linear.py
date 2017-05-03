from __future__ import print_function

import tensorflow as tf
import pandas as pd
import numpy
from collections import OrderedDict
import matplotlib.pyplot as plt
rng = numpy.random


# Parameters
learning_rate = 1.2
training_epochs = 500
display_step = 1

# Training Data
train_X = pd.read_csv('train.csv').iloc[0:141240, 117:131] # Using only 14 efatures (117~131)
train_Y = pd.read_csv('train.csv').iloc[0:141240, 131:132]
train_X = numpy.array(train_X)
train_Y = numpy.array(train_Y)

# train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
#                          7.042,10.791,5.313,7.997,5.654,9.27,3.1])
# train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
#                          2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]

# Cross Validation Data
test_X = pd.read_csv('train.csv').iloc[141241:188318, 117:131] # Using only 14 efatures (117~131)
test_Y = pd.read_csv('train.csv').iloc[141241:188318, 131:132]
test_X = numpy.array(test_X)
test_Y = numpy.array(test_Y)
# test_X = numpy.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
# test_Y = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03]

# tf Graph Input
X = tf.placeholder(tf.float32, [None, 14])
Y = tf.placeholder(tf.float32, [None, 1])


# Set model weights
# W = tf.Variable(tf.zeros([14, 1]), name="weight")
W = tf.get_variable(name="weight", shape=[14, 1], regularizer=tf.contrib.layers.l2_regularizer(0.9))
b = tf.Variable(tf.zeros([1]), name="bias")

# Construct a linear model
pred = tf.add(tf.matmul(X, W), b)

# Mean squared error
cost = tf.reduce_sum(tf.abs(pred-Y)) / (n_samples)
regularized_cost = cost # This loss needs to be minimized

# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(regularized_cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    train_error = []
    cv_error = []
    count = 0
    plt.ion()
    for epoch in range(training_epochs):
        # for (x, y) in zip(train_X, train_Y): # Comment out for now as we run full batch.
            # x = numpy.reshape(x, (1, 14))
            # y = numpy.reshape(y, (1, 1))
        sess.run(optimizer, feed_dict={X: train_X, Y: train_Y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(regularized_cost, feed_dict={X: train_X, Y:train_Y})
            print("!!!Epoch:", '%04d' % (epoch+1), "\n Training Cost=", "{:.9f}".format(c), \
                "\nW=", sess.run(W), "\nb=", sess.run(b))

            print("Testing... (Mean Absolute error Comparison)")
            testing_cost = sess.run(
                tf.reduce_sum(tf.abs(pred - Y)) / (test_X.shape[0]),
                feed_dict={X: test_X, Y: test_Y})  # same function as cost above
            print("Cross Validation Cost=", testing_cost)
            print("Mean Absolute Error Difference:", (testing_cost - c))
            train_error.append(c)
            cv_error.append(testing_cost)
            count = count + 1

            # Graphic display
            plt.plot(range(count), cv_error, '-r', label='CV Error')
            plt.plot(range(count), train_error, '-b', label='Training Error')
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            plt.pause(0.01)

    plt.ioff()
    _ = plt.plot(range(count), cv_error, '-r', label='CV Error')
    _ = plt.plot(range(count), train_error, '-b', label='Training Error')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    _ = plt.legend(by_label.values(), by_label.keys())
    _ = plt.show()

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Trying out Tensorboard
    writer = tf.summary.FileWriter("output", sess.graph)
	print(sess.run(h))
	writer.close()

    with tf.name_scope("MyOperationGroup"):
    with tf.name_scope("Scope_A"):
        a = tf.add(1, 2, name="Add_these_numbers")
        b = tf.multiply(a, 3)
    with tf.name_scope("Scope_B"):
        c = tf.add(4, 5, name="And_These_ones")
        d = tf.multiply(c, 6, name="Multiply_these_numbers")

    with tf.name_scope("Scope_C"):
        e = tf.multiply(4, 5, name="B_add")
        f = tf.div(c, 6, name="B_mul")
    g = tf.add(b, d)
    h = tf.multiply(g, f)



