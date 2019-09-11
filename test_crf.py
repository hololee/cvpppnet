import numpy as np
import method
import os
from scipy.misc import imread
import DataGen
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, create_pairwise_gaussian, softmax_to_unary

# get images.
label_images = []

dir = '/data1/LJH/cvpppnet/A1_predict'
all_file_list = os.listdir(dir)
all_file_list.sort()

for image_names in all_file_list:
    if ".py" not in image_names:
        real_path = dir + "/" + image_names
        # load images.
        label_images.append(imread(real_path, mode='L'))

# change to ndarray
label_images = np.array(label_images, dtype=np.float32)
label_images = np.expand_dims(label_images, 3)
label_images = np.where(label_images == 0, 0.0001, label_images)
label_images = label_images / np.amax(label_images)

# crf testing
dataG = DataGen.DataGen()
all_images = np.array(dataG.load_images())

h, w = dataG.getImageSize()

output_crfs = []

for i in range(len(all_images)):
    output_crf = method.dense_crf(img=all_images[i], probs=label_images[i], n_iters=3,
                                  sxy_gaussian=(42, 42), compat_gaussian=15,
                                  kernel_gaussian=dcrf.DIAG_KERNEL,
                                  normalisation_gaussian=dcrf.NORMALIZE_SYMMETRIC,
                                  sxy_bilateral=(42, 42), compat_bilateral=15,
                                  srgb_bilateral=(12, 12, 12),
                                  kernel_bilateral=dcrf.DIAG_KERNEL,
                                  normalisation_bilateral=dcrf.NORMALIZE_SYMMETRIC)

    fig = plt.figure()
    fig.set_size_inches(9, 3)  # 1800 x600

    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    ax1.set_title("origin")
    ax2.set_title("output")
    ax3.set_title("adjust CRF")

    ax1.imshow(all_images[i])
    ax2.imshow(label_images[i].squeeze(), cmap="jet")
    ax3.imshow(output_crf.squeeze(), cmap="jet")

    plt.show()

    output_crfs.append(output_crf)
