#!/bin/bash

tfpath='../tensorflow'
$tfpath/bazel-bin/tensorflow/contrib/lite/toco/toco \
    -input_format=TENSORFLOW_GRAPHDEF \
    --output_format=TFLITE \
    --input_file=./model/frozen_cnn_bler.pb \
    --output_file=./model/cnn_bler.tflite\
    --input_arrays=input \
    --output_arrays=inference \
    --input_shapes=1,28,28,3
