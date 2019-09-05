import tensorflow as tf

# convolution type.
TYPE_NORMAL = 'normal'
TYPE_ATROUS = 'atrous'

# activate functions.
FUNC_RELU = 'relu'
NONE = 'none'


def layers(type, input_map, tar_dim, name, act_func, batch_norm, pooling={'size': 2, 'stride': 2}):
    weight = tf.Variable(tf.random_normal([3, 3, input_map.get_shape().as_list()[3], tar_dim], stddev=0.1), name=name)
    bias = tf.Variable([0.1])

    # choose type
    if type == TYPE_NORMAL:
        conv_result = tf.nn.conv2d(input_map, weight, strides=[1, 1, 1, 1], padding="SAME") + bias
    elif type == TYPE_ATROUS:
        conv_result = tf.nn.atrous_conv2d(input_map, weight, rate=2, padding="VALID") + bias

    # activation
    if act_func == None:
        pass
    elif act_func == FUNC_RELU:
        conv_result = tf.nn.relu(conv_result, name + FUNC_RELU)

    # batch normalization
    if batch_norm.use_batch_norm:
        # using batch normalization.
        conv_result = tf.layers.batch_normalization(conv_result, center=True, scale=True, training=batch_norm.is_train)

    # max pooling.
    if pooling != None:
        conv_result = tf.nn.max_pool(conv_result, ksize=[1, pooling['size'], pooling['size'], 1],
                                     strides=[1, pooling['stride'], pooling['stride'], 1],
                                     padding='SAME')

    # print shape of array
    print(conv_result.shape)

    return conv_result


def bi_linear_interpolation(input_map, original_map_size=(530, 500)):
    conv_result = tf.image.resize_images(input_map, size=original_map_size,
                                         method=tf.image.ResizeMethod.BILINEAR)

    # print shape of array
    print(conv_result.shape)

    return conv_result
