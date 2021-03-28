

import numpy as np
import cv2 as cv 
import pdb
import re
import os
import nibabel as nib
import matplotlib.pyplot as plt
import utils

test_path =  r'C:\Users\Alexandre\Repertoire\SRM4BMRI\Intermediate_Loss\results_mri3'
original_path = r'C:\Users\Alexandre\Repertoire\SRM4BMRI\Intermediate_Loss\test_mri\1010.nii'
og_3d = nib.load(original_path)
ground_truth_img = og_3d.get_fdata()
ground_truth_img = ground_truth_img[1:,:,:]
number = "1010"

for f in os.listdir(test_path):
    if number in f:
        img_path = os.path.join(test_path,f)
        img_3d = nib.load(img_path)
        final_img = img_3d.get_fdata()
        print(final_img.shape,ground_truth_img.shape)
        print("File name: ",f)
        print("PSNR:",utils.psnr(final_img,ground_truth_img))
        print("SSIM:",utils.ssim(final_img,ground_truth_img))