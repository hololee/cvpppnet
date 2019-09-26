import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.misc import imread
import numpy as np
import cv2

test = imread("/data1/LJH/cvpppnet/A1_predict_edge/plant_out_epc150_1569487222140.png", mode='L')

value = 1

test[np.where(test < value)] = 0
test[np.where(test >= value)] = 255

img_sobel_x = cv2.Sobel(test, cv2.CV_64F, 1, 0, ksize=3)
img_sobel_x = cv2.convertScaleAbs(img_sobel_x)
img_sobel_y = cv2.Sobel(test, cv2.CV_64F, 0, 1, ksize=3)
img_sobel_y = cv2.convertScaleAbs(img_sobel_y)

img_sobel = cv2.addWeighted(img_sobel_x, 1, img_sobel_y, 1, 0)

plt.imshow(img_sobel)
plt.show()




# plt.imshow(test)
# plt.show()

