import tensorflow as tf
import numpy as np
import os
from scipy.misc import imread
import matplotlib.pyplot as plt
import scipy.misc
import cv2

dir1 = '/data1/LJH/cvpppnet/A1'
dir2 = '/data1/LJH/cvpppnet/A1_predict_enet'

all_img_list = os.listdir(dir1)
all_img_list.sort()

all_mask_list = os.listdir(dir2)
all_mask_list.sort()

rgbs_name = []
fgs_name = []

# slect origin images.

for titles in all_img_list:
    if "_rgb" in titles:
        rgbs_name.append(titles)

for titles in all_mask_list:
    fgs_name.append(titles)

for index in range(len(rgbs_name)):
    real_path = dir1 + "/" + rgbs_name[index]
    real_path2 = dir2 + "/" + fgs_name[index]
    # load images.
    rgb_images = imread(real_path, mode='RGB')
    fg_images = imread(real_path2, mode='L')

    rgb_images[np.where(fg_images < 5)] = 0
    scipy.misc.imsave("/data1/LJH/cvpppnet/A1_predict_enet_mask/result{0:03d}.png".format(index), np.squeeze(rgb_images))
