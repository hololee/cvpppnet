import tensorflow as tf
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
label_images = np.where(label_images == 0, 32, label_images)

for i in range(len(label_images)):
    result = tf.nn.softmax(label_images[i], dim=0)
    with tf.Session() as sess:
        soft = sess.run(result)

        plt.imshow(np.squeeze(soft))
        plt.show()
