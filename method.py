import tensorflow as tf
import numpy as np

FUNC_RELU = 'relu'
NO_ACTIVATION = 'none'

def conv_2d(input_map, tar_dim, name, act_func=FUNC_RELU, add_bn=True,):
    weight = tf.Variable(tf.random_normal([3, 3, input_map.shape().aslist()[2], tar_dim], stddev=0.1), name=name)
    bias = tf.Variable([0.1])

    conv_result = tf.nn.conv2d(input_map, weight, strides=[1, 1, 1, 1], padding="same") + bias

    # activation
    if act_func == NO_ACTIVATION:
        pass
    elif act_func == FUNC_RELU:
        conv_result = tf.nn.relu(conv_result, name + FUNC_RELU)

    # batch normalization
    if add_bn:
        # using batch normalization.
        tf.layers.batch_normalization(conv_result, center=True, scale=True, )

    return conv_result
