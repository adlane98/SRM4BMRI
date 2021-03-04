#  Copyright (c) 2020. Mariana-Iuliana Georgescu, Radu Tudor Ionescu, Nicolae Verga
#  Convolutional Neural Networks with  Intermediate Loss for 3D Super-Resolution of CT and MRI Scans
#  Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
#  (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#

import numpy as np
import cv2 as cv
from PIL import Image
import os
import glob
import numpy as np
from skimage.measure import compare_ssim as ssim_sk 
from skimage.measure import compare_psnr as psnr_sk 

import math 
import pdb

SHOW_IMAGES = False


def my_psnr(img1, img2):
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0 
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def psnr(img1, img2):  
    img1[img1 < 0] = 0
    img1[img1 > 255] = 255
    
    img2[img2 < 0] = 0
    img2[img2 > 255] = 255
    
    img1 = np.uint8(img1)
    img2 = np.uint8(img2)
    res = psnr_sk(img1, img2)
    if math.isinf(res):
        return 100
    else:
        return res


def ssim(img1, img2):
    img1[img1 < 0] = 0
    img1[img1 > 255] = 255

    img2[img2 < 0] = 0
    img2[img2 > 255] = 255

    img1 = np.uint8(img1)
    img2 = np.uint8(img2)
    if(img1.shape[2] == 1):
        return ssim_sk(np.squeeze(img1), np.squeeze(img2))
    return ssim_sk(img1, img2)
