import os
import argparse
import tensorflow as tf
from tensorflow.python.framework import graph_util


def freeze_graph():

    output_graph = "./checkpoint/frozen_inference_graph.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations 
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph('./checkpoint/ckp-486.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, './checkpoint/ckp-486')

        # We use a built-in TF helper to export variables to constants
        #constant_graph=graph_util.convert_variables_to_constants(sess,tf.get_default_graph().as_graph_def(),["SparseToDense"])
        
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
             ['Image', 'PixelLabel', 'segment/Sigmoid'])# The output node names are used to select the usefull nodes
        
        # Finally we serialize and dump the output graph to the filesystem
        #with tf.gfile.GFile(output_graph, mode='wb') as f:
	#    f.write(constant_graph.SerializeToString())
        #print("%d ops in the final graph." % len(constant_graph.node))
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

freeze_graph()