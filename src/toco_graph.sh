#!/bin/bash

tfpath='../tensorflow'
$tfpath/bazel-bin/tensorflow/contrib/lite/toco/toco \
    -input_format=TENSORFLOW_GRAPHDEF \
    --output_format=TFLITE \
    --input_file=./model2/frozen_cnn.pb \
    --output_file=./model2/cnn.tflite\
    --input_arrays=input \
    --output_arrays=inference \
    --input_shapes=1,28,28,3
