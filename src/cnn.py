#!/usr/bin/env python

import tensorflow as tf
import cv2
from pathlib import Path
import numpy as np
import model
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
import random

def transform_onehot(labels):
    labels = np.array(labels).reshape(1, -1)
    encoder = OneHotEncoder()
    ys = encoder.fit_transform(labels.T).toarray()
    return ys


def get_file_list(path, ext='*jpg'):
    path_obj = Path(path)
    files = list(path_obj.glob(ext))
    return files


def load_data(beautiful_path,
              blur_path,
              dark_path):

    # files
    beautiful_files = get_file_list(beautiful_path)
    blur_files = get_file_list(blur_path)
    dark_files = get_file_list(dark_path)

    # labels
    labels = [1 for _ in range(len(beautiful_files))] +\
             [2 for _ in range(len(blur_files))] +\
             [3 for _ in range(len(dark_files))]

    # one hot vector
    ys = transform_onehot(labels)

    # make dataset
    images = []
    files = beautiful_files + blur_files + dark_files
    for f_in in files:
        img = cv2.imread(str(f_in))
        img = cv2.resize(img, (28, 28))
        images.append(img.flatten().astype(np.float32) / 255.0)

    return np.array(images), ys


def fit(X, y, output_path,
        batch_size=50, verbose=False):

    with tf.Graph().as_default():

        # input/output
        _x = tf.placeholder(tf.float32, shape=[None, 2352], name='in')
        _y = tf.placeholder(tf.float32, shape=[None, 3])

        # train logits
        training_logits = model.convolutional(_x)

        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=_y,
                                                              logits=training_logits))
        # optimizer
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

        correct_prediction = tf.equal(tf.argmax(training_logits, 1), tf.argmax(_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        saver = tf.train.Saver()

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            for epoch in range(100):
                images, labels = shuffle(X, y)

                for i in range(0, int(len(images)), batch_size):
                    batch = X[i: i + batch_size], y[i: i + batch_size]
                    train_step.run(feed_dict={_x: batch[0], _y: batch[1]})

                if epoch % 10 == 0:
                    i = random.randint(0, int(len(images)/batch_size)) * batch_size
                    batch = X[i: i + batch_size], y[i: i + batch_size]
                    train_accuracy = accuracy.eval(feed_dict={
                        _x: batch[0], _y: batch[1]})

                    if verbose:
                        print('step %d, training accuracy %g' % (epoch, train_accuracy))

            save_path = saver.save(sess, f"{output_path}/cnn.ckpt")

            if verbose:
                print("Model saved in file: %s" % save_path)

            tf.train.write_graph(sess.graph_def, output_path,
                                 'cnn.pb', as_text=False)
            tf.train.write_graph(sess.graph_def, output_path,
                                 'cnn.pbtxt', as_text=True)


def main():
    images, labels = load_data(beautiful_path='./row_data/resized/',
                               blur_path='./row_data/blured/',
                               dark_path='./row_data/dark/')

    fit(images, labels, output_path='./model_tmp', verbose=True)


if __name__ == "__main__":
    main()
