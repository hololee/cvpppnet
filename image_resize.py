from PIL import Image
import os

dir1 = '/data1/LJH/cvpppnet/A3/'


all_img_list = os.listdir(dir1)
all_img_list.sort()


for item_path in all_img_list:
    image = Image.open(dir1 + item_path)
    resize_image = image.resize((512, 512))
    resize_image.save(dir1 + item_path, "PNG", quality=100)
