import tensorflow as tf
import pydensecrf.densecrf as dcrf
import numpy as np
import config_etc
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, create_pairwise_gaussian, softmax_to_unary
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral

# convolution type.
TYPE_NORMAL = 'normal'
TYPE_ATROUS = 'atrous'

# activate functions.
FUNC_RELU = 'relu'
NONE = 'none'


def layers(type, input_map, tar_dim, name, act_func, batch_norm, pooling={'size': 2, 'stride': 2}):
    weight = tf.Variable(
        tf.random_normal([3, 3, input_map.get_shape().as_list()[3], tar_dim], stddev=config_etc.TRAIN_STDDV), name=name)
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


# ====== crf test

def process_crf(image, predict, height, width, n_classes):
    d = dcrf.DenseCRF2D(width, height, n_classes)
    U = -np.log(predict)  # Unary potential.
    U = U.transpose(2, 1, 0).reshape((n_classes, -1))
    d.setUnaryEnergy(U)

    # if image is not None:
    #     assert (image.shape[0:2] == (height, width)), "The image height and width must coincide with dimensions of the logits."
    #
    #     # d.addPairwiseGaussian(sxy=(3, 3), compat=4,
    #     #                       kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    #     # d.addPairwiseBilateral(sxy=(45, 45), compat=10,
    #     #                        kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC,
    #     #                        srgb=(13, 13, 13), rgbim=image)
    #
    #     d.addPairwiseGaussian(sxy=44, compat=3)
    #     d.addPairwiseBilateral(sxy=25, srgb=3, rgbim=image, compat=3)

    Q = d.inference(5)
    preds = np.array(Q, dtype=np.float32)  # .reshape((n_classes, width, height)).transpose(2, 1, 0)
    map = np.argmax(Q, axis=0).reshape((width, height))

    return np.expand_dims(preds, 0)


def dense_crf(probs, n_classes=1, img=None, n_iters=3,
              sxy_gaussian=(1, 1), compat_gaussian=3,
              kernel_gaussian=dcrf.DIAG_KERNEL,
              normalisation_gaussian=dcrf.NORMALIZE_SYMMETRIC,
              sxy_bilateral=(24, 24), compat_bilateral=3,
              srgb_bilateral=(5, 5, 5),
              kernel_bilateral=dcrf.DIAG_KERNEL,
              normalisation_bilateral=dcrf.NORMALIZE_SYMMETRIC):
    """DenseCRF over unnormalised predictions.
       More details on the arguments at https://github.com/lucasb-eyer/pydensecrf.

    Args:
      probs: class probabilities per pixel.
      img: if given, the pairwise bilateral potential on raw RGB values will be computed.
      n_iters: number of iterations of MAP inference.
      sxy_gaussian: standard deviations for the location component of the colour-independent term.
      compat_gaussian: label compatibilities for the colour-independent term (can be a number, a 1D array, or a 2D array).
      kernel_gaussian: kernel precision matrix for the colour-independent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_gaussian: normalisation for the colour-independent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
      sxy_bilateral: standard deviations for the location component of the colour-dependent term.
      compat_bilateral: label compatibilities for the colour-dependent term (can be a number, a 1D array, or a 2D array).
      srgb_bilateral: standard deviations for the colour component of the colour-dependent term.
      kernel_bilateral: kernel precision matrix for the colour-dependent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_bilateral: normalisation for the colour-dependent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).

    Returns:
      Refined predictions after MAP inference.
    """
    h, w, _ = probs.shape

    probs = probs.transpose(2, 0, 1).copy(order='C')  # Need a contiguous array.

    d = dcrf.DenseCRF2D(w, h, n_classes)  # Define DenseCRF model.
    U = -np.log(probs)  # Unary potential.
    U = U.reshape((n_classes, -1))  # Needs to be flat.
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian,
                          kernel=kernel_gaussian, normalization=normalisation_gaussian)
    if img is not None:
        assert (img.shape[0:2] == (h, w)), "The image height and width must coincide with dimensions of the logits."
        d.addPairwiseBilateral(sxy=sxy_bilateral, compat=compat_bilateral,
                               kernel=kernel_bilateral, normalization=normalisation_bilateral,
                               srgb=srgb_bilateral, rgbim=img)
    Q = d.inference(n_iters)
    preds = np.array(Q, dtype=np.float32).reshape((n_classes, h, w)).transpose(1, 2, 0)
    return np.expand_dims(preds, 0)



