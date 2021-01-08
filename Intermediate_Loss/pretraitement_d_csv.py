import numpy as np
import random as rd
import cv2 as cv
import os
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt

# separate input image into patches
# pacthes are saved in two directories
# one for the groud truth
# the other one for the input of the network
def extract_patch_and_save(img, dim_p, stride, scale, folder_in, folder_gt, index,images):
    print("---"+images+"---"+str(index))
    gt_dics = []
    in_dics = []
    h = img.shape[0]
    w = img.shape[1]
    dsize = (dim_p//scale,dim_p)
    for i in range(0,h-dim_p,stride):
        for j in range(0,w-dim_p,stride):
            index += 1
            gt_patch = img[i:i+dim_p,j:j+dim_p]
            sigma = rd.uniform(0,1)*0.5
            if (rd.uniform(0,1) < 0.5):
                in_patch = cv.resize(cv.GaussianBlur(gt_patch,(3,3),sigma),dsize)
            else:
                in_patch = cv.resize(gt_patch,dsize)

            if (np.sum(in_patch) < 5):
                continue

            #cv.imwrite(folder_gt+'/'+str(index)+'.png',gt_patch)
            #cv.imwrite(folder_in+'/'+str(index)+'.png',in_patch)
            gt_dics += [{'shape':gt_patch.shape,'image_1d':gt_patch.flatten().tolist()}]
            in_dics += [{'shape':in_patch.shape,'image_1d':in_patch.flatten().tolist()}]
    if gt_dics != []:
        df_gt = pd.DataFrame.from_dict(gt_dics)
        df_in = pd.DataFrame.from_dict(in_dics)

        df_gt.to_csv(folder_gt+'/'+images+'-'+str(index)+'.csv')
        df_in.to_csv(folder_in+'/'+images+'-'+str(index)+'.csv')



    return index

# **************************************************
#             PATCHES EXTRACTION TEST
# file = r'C:\Users\furet\Pictures\image.jpg'
# img = cv.imread(file)
# inp = r'C:\Users\furet\Pictures\inputs'
# gt = r'C:\Users\furet\Pictures\truth'
# print(extract_patch_and_save(img,64,30,2,inp,gt,0))
#
# **************************************************


# **************************************************
#                      MAIN
# Params
# main_dir = r'D:\Utilisateurs\Alexandre\Repertoire_D\projet_super_resolution\data\train\train_data\7T'
main_dir = r'D:\Utilisateurs\Alexandre\Repertoire_D\projet_super_resolution\data\marmoset_train_d_x4\train_data\3T'
dim_patch = 28
stride = 7
scale = 2
f_in = r'D:\Utilisateurs\Alexandre\Repertoire_D\projet_super_resolution\data\marmoset_train_d_x4\train_data\inputs'
f_gt = r'D:\Utilisateurs\Alexandre\Repertoire_D\projet_super_resolution\data\marmoset_train_d_x4\train_data\ground_truth'
index = 0

for images in os.listdir(main_dir):
    imgs = nib.load(main_dir+'\\'+images)
    fdata = imgs.get_fdata()
    if (len(imgs.shape) != 3):
        print('Input image is not T1 weighted.\nSkipping this image...')
        continue
    for i in range(imgs.shape[0]):
        index = extract_patch_and_save(fdata[i,:,:],dim_patch,stride,scale,f_in,f_gt,index,images)

    # for i in range(fdata.shape[-1]):
    #     index = extract_patch_and_save(fdata[:,:,i],dim_patch,stride,scale,f_in,f_gt,index, images)

# **************************************************
