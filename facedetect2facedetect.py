import tensorflow as tf
import sys
from tensorflow.core.framework import graph_pb2
import copy

import numpy as np


# load our graph
def load_graph(filename):
    graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(filename, 'rb') as f:
        graph_def.ParseFromString(f.read())
    return graph_def
graph_def = load_graph('./models/face/opencv_face_detector_uint8.pb')

constants0 = {'data_bn/beta:0': 3, 'conv1_bn_h/beta:0': 32, 'layer_128_1_bn1_h/beta:0': 32,
              'layer_256_1_bn1/beta:0': 128, 'layer_512_1_bn1/beta:0': 256, 'last_bn_h/beta:0': 256}
input_names = ['data:0']
inputs = {}
for k, v in constants0.items():
    k = k[:-2]
    print('Add: ',k)
    input_names.append(k)
    inputs[k] = tf.constant(0,tf.float32,shape=[v],name=k)

new_graph_def = graph_pb2.GraphDef()
for node in graph_def.node:
    if inputs.get(node.name,None) is not None:
        print('replace: ',node.name)
        c = inputs.get(node.name)
        new_graph_def.node.extend([c.op.node_def])
    else:
        new_graph_def.node.extend([copy.deepcopy(node)])

# save new graph
with tf.gfile.GFile('./models/face/opencv_face_detector_uint8-fixed.pb', "wb") as f:
    f.write(new_graph_def.SerializeToString())