from scipy.misc import imread, imsave, imresize
import numpy as np
import os


class DataGen:

    def __init__(self):
        self.dir = '/data1/LJH/cvpppnet/A1'
        all_file_list = os.listdir(self.dir)
        all_file_list.sort()

        self.rgbs = []
        self.fgs = []

        self.rgb_images = []
        self.fg_images = []

        # slect origin images.

        for titles in all_file_list:
            if '_rgb' in titles:
                print(titles + ": color image")
                # color images
                self.rgbs.append(titles)

            if '_fg' in titles:
                print(titles + ": foreground")
                # foregrounds
                self.fgs.append(titles)

    def load_images(self):
        for image_names in self.rgbs:
            real_path = self.dir + "/" + image_names
            # load images.
            self.rgb_images.append(imread(real_path, mode='RGB'))

        return self.rgb_images

    def load_foregrounds(self):
        for image_names in self.fgs:
            real_path = self.dir + "/" + image_names
            # load images.
            self.fg_images.append(imread(real_path, mode='L'))

        return self.fg_images
