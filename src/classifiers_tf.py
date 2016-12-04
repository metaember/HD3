# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

# Import data
# from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
FLAGS = None

# dataset = 2uple || dataset[0] = input in R_nbInput,dimensionInput || dataset[1] = labels in R_nbInput,dimensionOutput
def classify_with_softmax_nn(dataset, percentage_train, percentage_test, batch_size, learning_rate):

    # Create the model
    x = tf.placeholder(tf.float32, [None, dataset[0].shape[1]])
    W = tf.Variable(tf.zeros([dataset[0].shape[1], dataset[1].shape[1]]))
    b = tf.Variable(tf.zeros([dataset[1].shape[1]]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, dataset[1].shape[1]])

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    sess = tf.InteractiveSession()

    # Basic stuff to compute batches indices
    nb_batches = int(dataset[0].shape[0] * percentage_train // batch_size)

    start_index_test = nb_batches * batch_size
    end_index_test = start_index_test + int(dataset[0].shape[0] * percentage_test)

    # Train
    tf.global_variables_initializer().run()
    for i in range(nb_batches):
        batch_xs, batch_ys = dataset[0][100*i:100*(i+1), :], dataset[1][100*i:100*(i+1), :]
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    batch_xs_test = dataset[0][start_index_test:end_index_test, :]
    batch_ys_test = dataset[1][start_index_test:end_index_test, :]

    return sess.run(accuracy, feed_dict={x: batch_xs_test, y_: batch_ys_test})
