#!/usr/bin/env python

import tensorflow as tf
import model
import cv2
import glob
import numpy as np
import model
from tensorflow.python.framework import graph_util
from tensorflow.python.tools.freeze_graph import freeze_graph
from sklearn import utils


def load_data(true_path, false_path):
    true_files = glob.glob(true_path + '*jpg')
    false_files = glob.glob(false_path + '*jpg')

    labels = np.append(np.ones(len(true_files)),
                       np.zeros(len(false_files)))

    labels = labels.reshape(-1, 1)

    images = []
    files = true_files + false_files
    for f_in in files:
        img = cv2.imread(f_in)
        img = cv2.resize(img, (28, 28))
        images.append(img.flatten().astype(np.float32) / 255.0)

    return np.array(images), labels


def main():

    # load data
    images, labels = load_data(true_path='./resized/', false_path='./blured/')
    batch_size = 50

    with tf.Graph().as_default():

        # input/output
        x = tf.placeholder(tf.float32, shape=[None, 2352], name='in')
        y = tf.placeholder(tf.float32, shape=[None, 1])

        # train logits
        training_logits = model.convolutional(x)

        loss = tf.losses.mean_squared_error(
            labels=y, predictions=training_logits)

        # optimizer
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

        correct_prediction = tf.equal(tf.round(training_logits), y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        saver = tf.train.Saver()

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            for epoch in range(100):
                images, labels = utils.shuffle(images, labels)

                for i in range(0, int(len(images)), batch_size):

                    batch = images[i: i +
                                   batch_size], labels[i: i + batch_size]

                    train_step.run(
                        feed_dict={x: batch[0], y: batch[1]})

                if epoch % 10 == 0:
                    train_accuracy = accuracy.eval(feed_dict={
                        x: batch[0], y: batch[1]})
                    print('step %d, training accuracy %g' %
                          (epoch, train_accuracy))

            save_path = saver.save(sess, "model/cnn_bler.ckpt")
            print("Model saved in file: %s" % save_path)

            tf.train.write_graph(sess.graph_def, './model',
                                 'cnn_bler.pb', as_text=False)
            tf.train.write_graph(sess.graph_def, './model',
                                 'cnn_bler.pbtxt', as_text=True)


if __name__ == "__main__":
    main()
