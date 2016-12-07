import tensorflow as tf
import numpy as np


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




def classify_with_conv_nn(dataset, percentage_train, percentage_test, batch_size, learning_rate, training_iterations, verbose):



        # Create the model
        x = tf.placeholder(tf.float32, [None, dataset[0].shape[1]])
        W = tf.Variable(tf.zeros([dataset[0].shape[1], dataset[1].shape[1]]))
        b = tf.Variable(tf.zeros([dataset[1].shape[1]]))
        y = tf.matmul(x, W) + b

        y_ = tf.placeholder(tf.float32, [None, dataset[1].shape[1]])


        # Create session
        sess = tf.InteractiveSession()

        # Some utility funcitons

        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME')


        # Layer 1
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        x_image = tf.reshape(x, [-1,28,28,1])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        # Layer 2
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        # Densely connected layer
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # dropout
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # readout layer
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        # globals
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # run initialisation of global vars
        sess.run(tf.global_variables_initializer())

        # Basic stuff to compute batches indices
        nb_batches = int(dataset[0].shape[0] * percentage_train // batch_size)
        start_index_test = nb_batches * batch_size
        end_index_test = start_index_test + int(dataset[0].shape[0] * percentage_test)

        images = dataset[0]
        labels = dataset[1]
        epochs_completed = 0
        index_in_epoch = 0
        num_examples = images.shape[0]

        def next_batch(batch_size):
            """
            Return the next `batch_size` examples from this data set.
            Shuffles Data if we request more that what there is.
            """
            nonlocal index_in_epoch, images, labels, epochs_completed, num_examples
            start = index_in_epoch
            index_in_epoch += batch_size
            if index_in_epoch > num_examples:
                # Finished epoch
                epochs_completed += 1
                # Shuffle the data
                perm = np.arange(num_examples)
                np.random.shuffle(perm)
                images = images[perm]
                labels = labels[perm]
                # Start next epoch
                start = 0
                index_in_epoch = batch_size
                assert batch_size <= num_examples
            end = index_in_epoch
            return images[start:end], labels[start:end]


        for i in range(training_iterations):
            batch = next_batch(batch_size)
            if i%100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x:batch[0], y_: batch[1], keep_prob: 1.0})
                if verbose:
                    print("step {} of {} ({}%), training accuracy {:.4f}".format(i, training_iterations,
                            int(100*i/training_iterations), train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


        batch_xs_test = dataset[0][start_index_test:end_index_test, :]
        batch_ys_test = dataset[1][start_index_test:end_index_test, :]
        #print("test accuracy {}".format(accuracy.eval(feed_dict={ x: batch_xs_test, y_: batch_ys_test, keep_prob: 1.0})))

        return accuracy.eval(feed_dict={x: batch_xs_test, y_: batch_ys_test, keep_prob: 1.0})
