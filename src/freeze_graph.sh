#!/bin/bash

tfpath='../tensorflow'
$tfpath/bazel-bin/tensorflow/python/tools/freeze_graph \
    --input_graph=./model2/cnn.pb \
    --input_checkpoint=./model2/cnn.ckpt \
    --input_binary=true \
    --output_node_names=inference \
    --output_graph=./model2/frozen_cnn.pb
