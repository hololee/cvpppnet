import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import method
from batch_norm import BatchNorm

from DataGen import DataGen

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def get_shape(text, input):
    sess = tf.InteractiveSession()

    print("{}shape : {}".format(text, sess.run(tf.shape(input))))
    sess.close()


dataG = DataGen()

# TODO loaded images numpy array.
rgb_images = np.array(dataG.load_images())
fg_images = np.array(dataG.load_foregrounds())

# reshape
fg_images = np.reshape(fg_images, [fg_images.shape[0], fg_images.shape[1], fg_images.shape[2], 1])

print("rgb_images : " + str(np.shape(rgb_images)))
print("fg_images : " + str(np.shape(fg_images)))

# index = np.random.randint(0, len(rgb_images))
#
# plt.imshow(rgb_images[index])
# plt.show()
# plt.imshow(fg_images[index])
# plt.show()


########### holders

input_data = tf.placeholder(tf.float32,
                            shape=(None, np.shape(rgb_images)[1], np.shape(rgb_images)[2], np.shape(rgb_images)[3]))
ground_truth = tf.placeholder(tf.float32,
                              shape=(None, np.shape(fg_images)[1], np.shape(fg_images)[2], np.shape(fg_images)[3]))

is_train = tf.placeholder(tf.bool)

########### layer
# first step
# get_shape("input data: {}", input_data)
gen_convolution = method.layers(method.TYPE_NORMAL, input_data, 64, "layer1", method.FUNC_RELU,
                                BatchNorm(is_train=is_train, use_batch_norm=True), pooling=None)

gen_convolution = method.layers(method.TYPE_NORMAL, gen_convolution, 64, "layer2_pooling", method.FUNC_RELU,
                                BatchNorm(is_train=is_train, use_batch_norm=True), pooling={'size': 2, 'stride': 2})

# step 2
gen_convolution = method.layers(method.TYPE_NORMAL, gen_convolution, 128, "layer3", method.FUNC_RELU,
                                BatchNorm(is_train=is_train, use_batch_norm=True), pooling=None)
gen_convolution = method.layers(method.TYPE_NORMAL, gen_convolution, 128, "layer4_pooling", method.FUNC_RELU,
                                BatchNorm(is_train=is_train, use_batch_norm=True), pooling={'size': 2, 'stride': 2})

# step 3
gen_convolution = method.layers(method.TYPE_NORMAL, gen_convolution, 256, "layer5", method.FUNC_RELU,
                                BatchNorm(is_train=is_train, use_batch_norm=True), pooling=None)
gen_convolution = method.layers(method.TYPE_NORMAL, gen_convolution, 256, "layer6", method.FUNC_RELU,
                                BatchNorm(is_train=is_train, use_batch_norm=True), pooling=None)
gen_convolution = method.layers(method.TYPE_NORMAL, gen_convolution, 256, "layer7_pooling", method.FUNC_RELU,
                                BatchNorm(is_train=is_train, use_batch_norm=True), pooling={'size': 2, 'stride': 2})

# step 3
gen_convolution = method.layers(method.TYPE_NORMAL, gen_convolution, 512, "layer8", method.FUNC_RELU,
                                BatchNorm(is_train=is_train, use_batch_norm=True), pooling=None)
gen_convolution = method.layers(method.TYPE_NORMAL, gen_convolution, 512, "layer9", method.FUNC_RELU,
                                BatchNorm(is_train=is_train, use_batch_norm=True), pooling=None)
gen_convolution = method.layers(method.TYPE_ATROUS, gen_convolution, 512, "layer10_atrous", method.FUNC_RELU,
                                BatchNorm(is_train=is_train, use_batch_norm=True), pooling=None)
gen_convolution = method.layers(method.TYPE_NORMAL, gen_convolution, 512, "layer11", method.FUNC_RELU,
                                BatchNorm(is_train=is_train, use_batch_norm=True), pooling=None)
gen_convolution = method.layers(method.TYPE_NORMAL, gen_convolution, 512, "layer12", method.FUNC_RELU,
                                BatchNorm(is_train=is_train, use_batch_norm=True), pooling=None)
gen_convolution = method.layers(method.TYPE_ATROUS, gen_convolution, 512, "layer13_atrous", method.FUNC_RELU,
                                BatchNorm(is_train=is_train, use_batch_norm=True), pooling=None)

# for target one
gen_convolution = method.layers(method.TYPE_NORMAL, gen_convolution, 1, "layer13_pooling", method.NONE,
                                BatchNorm(is_train=is_train, use_batch_norm=False), pooling=None)

# bi interpolation, to original size.
gen_convolution = method.bi_linear_interpolation(gen_convolution)
