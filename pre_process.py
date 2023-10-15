import gzip
import nibabel as nib
from matplotlib import pylab as plt
import numpy as np
import copy
import os
import cv2


def get_name(root, mode_folder=True):
    for root, dirs, file in os.walk(root):
        if mode_folder:
            return sorted(dirs)
        else:
            return sorted(file)


def un_gz(file_name):
    f_name = file_name.replace(".gz", "")
    g_file = gzip.GzipFile(file_name)
    open(f_name, "wb+").write(g_file.read())
    g_file.close()


root_list = ["train/HGG", "train/LGG"]
for root_ in root_list:
    root = "/home/Ysh_data/Domain2/" + root_
    type_ = ["t1", "t2", "t1ce", "flair", "seg"]
    all_roots = get_name(root)
    for fold in all_roots:
        print(fold)
        os.makedirs(os.path.join(root, fold, "extracted"), exist_ok=True)
        for item in type_:
            current_gz_file = os.path.join(root, fold, fold + "_" + item + ".nii.gz")
            # un_gz(current_gz_file)
            img_name = os.path.join(root, fold, fold + "_" + item + ".nii")
            img = nib.load(img_name)
            h, w, n = img.dataobj.shape
            for i in range(n):
                if i < 20:
                    continue
                img_arr = img.dataobj[:, :, i]
                h, w = img_arr.shape
                fig, ax = plt.subplots()
                fig.set_size_inches(w / 100.0, h / 100.0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.axis("off")
                plt.imshow(img_arr, cmap="gray")
                plt.savefig(os.path.join(root, fold, "extracted", fold + "_" + item + "_" + str(i) + ".png"))
                plt.close()

# root2 = "/home/wangbowen/DATA/Domain2/train/HGG/Brats18_CBICA_AAB_1/Brats18_CBICA_AAB_1_t1.nii"
#
# img = nib.load(root2)
# print(img)
# print(img.dataobj.shape)
#
# queue = 155
# x = int((queue/10)**0.5) + 1
# num = 1
#
# for i in range(0, queue, 10):
#     img_arr = copy.deepcopy(img.dataobj[:, :, i])
#     # img_arr[img_arr > 0] = 1
#     plt.subplot(x, x, num)
#     plt.imshow(img_arr, cmap="gray")
#     num += 1
# plt.show()