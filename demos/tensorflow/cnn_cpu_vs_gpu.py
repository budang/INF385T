# code adapted from https://github.com/ageron/handson-ml/blob/master/13_convolutional_neural_networks.ipynb

import argparse
import numpy as np
import tensorflow as tf
import time

from tensorflow.examples.tutorials.mnist import input_data

def main(mode):

    ############################
    # (down)load MNIST dataset #
    ############################

    print("[INFO] Loading MNIST dataset")
    mnist = input_data.read_data_sets("/tmp/data/")

    print()

    ####################################
    # build CNN architecture and graph #
    ####################################

    print("[INFO] Building CNN architecture")

    # image dimensions
    height = 28
    width = 28
    channels = 1

    # inputs
    with tf.name_scope("inputs"):
      X = tf.placeholder(tf.float32, shape=[None, height * width * channels], name="X")
      X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
      y = tf.placeholder(tf.int32, shape=[None], name="y")

    # convolution layers
    conv1 = tf.layers.conv2d(X_reshaped, filters=32, kernel_size=3,
                             strides=1, padding='SAME',
                             activation=tf.nn.relu, name="conv1")
    conv2 = tf.layers.conv2d(conv1, filters=64, kernel_size=3,
                             strides=2, padding='SAME',
                             activation=tf.nn.relu, name="conv2")

    # pooling layer
    with tf.name_scope("pool3"):
        pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
        pool3_flat = tf.reshape(pool3, shape=[-1, 64 * 7 * 7]) # pool3_fmaps = conv2_fmaps

    # fully connected layer
    with tf.name_scope("fc1"):
        fc1 = tf.layers.dense(pool3_flat, 64, activation=tf.nn.relu, name="fc1")

    # output layer
    with tf.name_scope("output"):
        logits = tf.layers.dense(fc1, 10, name="output")
        Y_proba = tf.nn.softmax(logits, name="Y_proba")

    # training with cross-entropy loss function and Adam optimizer
    with tf.name_scope("train"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
        loss = tf.reduce_mean(xentropy)
        optimizer = tf.train.AdamOptimizer()
        training_op = optimizer.minimize(loss)

    # evaluation (accuracy)
    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # network initializer and saver
    with tf.name_scope("init_and_save"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    print()

    ###############################
    # train CNN on the CPU or GPU #
    ###############################

    n_epochs = 10
    batch_size = 100

    # set to use CPU
    if mode == 'cpu':
        device_count = 0
    else:
        device_count = 1
    config = tf.ConfigProto(device_count={"GPU": device_count})

    # train using batch gradient descent
    print("[INFO] Training the CNN on the " + mode.upper())
    with tf.Session(config=config) as sess:
        start = time.time()

        init.run()
        for epoch in range(n_epochs):
            for iteration in range(mnist.train.num_examples // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
            print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
        
            save_path = saver.save(sess, "./trained/my_mnist_model")

        end = time.time()

        print("[INFO] Model saved to ./trained/my_mnist_model.*")
        print("[INFO] Total training time: " + str((end - start) / 60) + " minutes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN demo on CPU vs GPU")
    parser.add_argument('-m', '--mode', required=True,
                        choices=["cpu", "gpu"],
                        nargs=1, action="store",
                        type=str, dest="mode",
                        help="Whether to use the CPU or GPU")

    args = vars(parser.parse_args())
    mode = args["mode"][0]

    main(mode)
