import os
import method
import matplotlib.pyplot as plt

# original images path list.
original_images_path = []

# get original images from A1 folder.
img_dir = '/data1/LJH/cvpppnet/A1'
all_file_list = os.listdir(img_dir)
all_file_list.sort()

for titles in all_file_list:
    if '_rgb' in titles:
        # color images
        original_images_path.append(img_dir + "/" + titles)

# result of segmentation path.
segmentation_images_path = []

seg_dir = "/data1/LJH/cvpppnet/A1_predict"
all_seg_list = os.listdir(seg_dir)
all_seg_list.sort()

for titles in all_seg_list:
    if 'output' in titles:
        segmentation_images_path.append(seg_dir + "/" + titles)

# save image path.
sv_dir = "/data1/LJH/cvpppnet/A1_predict_crf"

for idx, path in enumerate(original_images_path):
    print(segmentation_images_path[idx])
    result_image = method.apply_crf(original_image_path=path, output_image_path=segmentation_images_path[idx],
                                    final_result_path=sv_dir + "/crf_result{0:03d}.png".format(idx))

    # plt.imshow(result_image)
    # plt.show()
