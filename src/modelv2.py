# !/usr/bin/env python

import tensorflow as tf
import numpy as np
import pathlib
import cv2
from sklearn.preprocessing import OneHotEncoder


def convolutional():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape((28, 28, 3), input_shape=(2352,), name='input'),
        tf.keras.layers.Conv2D(filters=32,
                               kernel_size=[5, 5],
                               padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2),
        tf.keras.layers.Conv2D(filters=32,
                               kernel_size=[5, 5],
                               padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=1024, activation='relu'),
        tf.keras.layers.Dropout(rate=0.4, trainable=True),
        tf.keras.layers.Dense(units=3, activation='softmax', name='output')
    ])

    return model


def transform_onehot(labels):
    labels = np.array(labels).reshape(1, -1)

    encoder = OneHotEncoder()
    ys = encoder.fit_transform(labels.T).toarray()
    return ys


def get_file_list(path, ext='*jpg'):
    path_obj = pathlib.Path(path)
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

    # make dataset
    images = []
    files = beautiful_files + blur_files + dark_files
    for f_in in files:
        img = cv2.imread(str(f_in))
        img = cv2.resize(img, (28, 28))
        images.append(img.flatten().astype(np.float32) / 255.0)

    return np.array(images), labels


def main():

    images, labels = load_data(beautiful_path='./row_data/resized/',
                               blur_path='./row_data/blured/',
                               dark_path='./row_data/dark/')

    y = transform_onehot(labels)

    model = convolutional()
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(images, y, epochs=100, batch_size=50, verbose=1)

    model.summary()
    print(model.input)
    print(model.output)
    keras_file = "model_keras/cnn_model.h5"
    model.save(keras_file)
    #tf.keras.models.save_model(model, keras_file)

    converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file(keras_file,
                                                                      input_arrays=['input_input'],
                                                                      output_arrays=['output/Softmax'],
                                                                      input_shapes={'input_input': [3, 2352]})
    tflite_model = converter.convert()
    open("model_keras/converted_model.tflite", "wb").write(tflite_model)


if __name__ == '__main__':
    main()