#!/usr/bin/env python

import tensorflow as tf
import cv2
from pathlib import Path
import numpy as np
import sklearn
import model


def load_data(true_path, false_path):

    ptrue = Path(true_path)
    true_files = list(ptrue.glob('*.jpg'))

    pfalse = Path(false_path)
    false_files = list(pfalse.glob('*jpg'))

    labels = np.append(np.ones(len(true_files)),
                       np.zeros(len(false_files)))

    labels = labels.reshape(-1, 1)

    images = []
    files = true_files + false_files
    for f_in in files:

        img = cv2.imread(str(f_in))
        img = cv2.resize(img, (28, 28))
        images.append(img.flatten().astype(np.float32) / 255.0)

    return np.array(images), labels


def fit(X, y, output_path,
        batch_size =50, verbose=False):

    with tf.Graph().as_default():

        # input/output
        _x = tf.placeholder(tf.float32, shape=[None, 2352], name='in')
        _y = tf.placeholder(tf.float32, shape=[None, 1])

        # train logits
        training_logits = model.convolutional(_x)

        loss = tf.losses.mean_squared_error(
            labels=_y, predictions=training_logits)

        # optimizer
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

        correct_prediction = tf.equal(tf.round(training_logits), _y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        saver = tf.train.Saver()

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            for epoch in range(100):
                images, labels = sklearn.utils.shuffle(X, y)

                for i in range(0, int(len(images)), batch_size):

                    batch = X[i: i + batch_size], y[i: i + batch_size]
                    train_step.run(feed_dict={_x: batch[0], _y: batch[1]})

                if epoch % 10 == 0:
                    train_accuracy = accuracy.eval(feed_dict={
                        _x: batch[0], _y: batch[1]})

                    if verbose:
                        print('step %d, training accuracy %g' % (epoch, train_accuracy))

            save_path = saver.save(sess, f"{output_path}/cnn_bler.ckpt")

            if verbose:
                print("Model saved in file: %s" % save_path)

            tf.train.write_graph(sess.graph_def, output_path,
                                 'cnn_bler.pb', as_text=False)
            tf.train.write_graph(sess.graph_def, output_path,
                                 'cnn_bler.pbtxt', as_text=True)


def main():
    images, labels = load_data(true_path='./resized/', false_path='./blured/')
    fit(images, labels, output_path='./model_tmp', verbose=True)


if __name__ == "__main__":
    main()
