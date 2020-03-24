import cv2
import matplotlib.pyplot as plt
from skimage import color

img = cv2.imread("/data1/LJH/cvpppnet/A3/plant006_rgb.png")

img_sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
img_sobel_x = cv2.convertScaleAbs(img_sobel_x)
img_sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
img_sobel_y = cv2.convertScaleAbs(img_sobel_y)

img_sobel = cv2.addWeighted(img_sobel_x, 1, img_sobel_y, 1, 0)
img = color.rgb2gray(img_sobel) * 255

plt.imshow(img)
plt.show()
