#!/usr/bin/env python

import tensorflow as tf
import model
from tensorflow.python.framework import graph_util
from tensorflow.python.tools.freeze_graph import freeze_graph


def main():

    tf.reset_default_graph()

    with tf.Graph().as_default() as graph:

        # print_graph_nodes(graph_def)

        with tf.Session() as sess:

            x = tf.placeholder(tf.float32, shape=[None, 2352])
            #keep_prob = tf.placeholder("float")
            training_logits = model.convolutional(x)

            saver = tf.train.import_meta_graph(
                './model/cnn_bler.ckpt.meta', clear_devices=True)

            saver.restore(sess, "./model/cnn_bler.ckpt")

            graph_def = graph_util.convert_variables_to_constants(
                sess, graph.as_graph_def(), ["inference_1"])

            tf.train.write_graph(graph_def, '.',
                                 "model/cnn_bler.pb", as_text=False)

            # freeze_graph(input_graph="model/cnn_bler.pb",
            #             input_saver="",
            #             restore_op_name='save/restore_all',
            #             filename_tensor_name='save/Const:0',
            #             initializer_nodes='',
            #             input_binary=True,
            #             input_checkpoint="./model/cnn_bler.ckpt",
            #             output_node_names="inference_1",
            #             # restore_op_name='save/restore_all',
            #             # filename_tensor_name='save/Const:0',
            #             output_graph="model/frozen_cnn_bler.pb",
            #
            #             clear_devices=True)


if __name__ == "__main__":
    main()
