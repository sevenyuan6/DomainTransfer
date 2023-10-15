import cv2
from PIL import Image
from utils.vis_tools import show_single
import numpy as np
from data.data_utils import crop_coordinate


img_dir = "/home/wangbowen/DATA/Domain/REFUGE/Training400/Glaucoma/g0001.jpg"
mask_dir = "/home/wangbowen/DATA/Domain/REFUGE/Annotation-Training400/Disc_Cup_Masks/Glaucoma/g0001.bmp"

img = Image.open(img_dir).convert('RGB')
mask = Image.open(mask_dir)
start_x, end_x, start_y, end_y = crop_coordinate(mask)
print(start_x, end_x, start_y, end_y)

mask = np.array(mask)
img = np.array(img)
cropMask = mask[start_y:end_y, start_x:end_x]
cropImg = img[start_y:end_y, start_x:end_x]
print(cropImg.shape)


show_single(cropMask, save=False)
show_single(mask, save=False)
show_single(cropImg, save=False)
show_single(img, save=False)
