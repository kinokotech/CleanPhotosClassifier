#!/bin/bash

#tfpath=`pip show tensorflow | grep "Location: \(.\+\)$" | sed 's/Location: //'`
#echo $tfpath
tfpath='../tensorflow'
$tfpath/bazel-bin/tensorflow/python/tools/freeze_graph \
    --input_graph=./model/cnn_bler.pb \
    --input_checkpoint=./model/cnn_bler.ckpt \
    --input_binary=true \
    --output_node_names=inference \
    --output_graph=./model/frozen_cnn_bler.pb
