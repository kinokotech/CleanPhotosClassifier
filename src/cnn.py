#!/usr/bin/env python

import tensorflow as tf
import model
import cv2
import glob
import numpy as np
import model
from tensorflow.python.framework import graph_util


def load_data(true_path, false_path):
    true_files = glob.glob(true_path + '*jpg')
    false_files = glob.glob(false_path + '*jpg')

    labels = np.append(np.ones(len(true_files)),
                       -1 * np.ones(len(false_files)))

    labels = labels.reshape(-1, 1)

    images = []
    for f_in in true_files + false_files:
        img = cv2.imread(f_in)
        img = cv2.resize(img, (28, 28))
        images.append(img.flatten().astype(np.float32) / 255.0)

    return np.array(images), labels


def main():

    # load data
    images, labels = load_data(true_path='./resized/', false_path='./blured/')

    # input/output
    x = tf.placeholder(tf.float32, shape=[None, 2352])
    y = tf.placeholder(tf.float32, shape=[None, 1])

    # train logits
    training_logits = model.convolutional(x)

    # loss function
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=training_logits))

    # optimizer
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(
        tf.argmax(training_logits, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    batch_size = 50

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(100):
            for i in range(int(len(images) / batch_size)):
                batch = images[i: i + batch_size], labels[i: i + batch_size]

                train_step.run(
                    feed_dict={x: batch[0], y: batch[1]})

            if epoch % 10 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y: batch[1]})
                print('step %d, training accuracy %g' %
                      (epoch, train_accuracy))

        save_path = saver.save(sess, "model/cnn_bler.ckpt")
        print("Model saved in file: %s" % save_path)

        converter = tf.contrib.lite.TocoConverter.from_session(sess, [
                                                               x], [training_logits])
        tflite_model = converter.convert()
        open("model/cnn_bler.tflite", "wb").write(tflite_model)


if __name__ == "__main__":
    main()
