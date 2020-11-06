
#  Copyright (c) 2020. Mariana-Iuliana Georgescu, Radu Tudor Ionescu, Nicolae Verga
#  Convolutional Neural Networks with  Intermediate Loss for 3D Super-Resolution of CT and MRI Scans
#  Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
#  (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#

import numpy as np
import utils
import params
from sklearn.utils import shuffle
import cv2 as cv 
import random 
import pdb
import os
import glob
import pandas as pd 

# Lis les patchs dans un csv
def get_image_from_csv(df):
    
    # Les tableaux en lectures de csv sont nativement en string, il faut les convertir
    df['image_1df'] = [[float(s) for s in df['image_1d'].iloc[i][1:-1].split(',')] for i in range(df.shape[0])]
    size = [int(s) for s in df['shape'][0][1:-1].split(',')]
    # list-> array 1d
    df['image_1dfa'] = [np.array(df['image_1df'][i]) for i in range(df.shape[0])]
    # array 2d
    df['image'] = [ df['image_1dfa'][i].reshape(size) for i in range(df.shape[0])]
    res = [ np.expand_dims(df['image'][i],2) for i in range(df.shape[0])] #df['image'].array
    res = np.array(res)
    return res

# modification de la fonction de base utils.py
def read_all_patches_from_directory_csv(base_dir, folder='', return_np_array=True):
    '''
        This function reads the images from the base_dir (walk in every dir named folder and read images).
        The output is list with nd-array (num_images, height, width, channels) and the minimum btw the min height and min width.
    '''
    if not os.path.exists(base_dir):
        print('Error!! Folder base name does not exit')
        
    images = [] 
    folder_names = os.listdir(base_dir) 
    print(base_dir,folder_names)
    for folder_name in folder_names:      
        
        images_path = os.path.join(base_dir, folder_name, folder, '*' + "csv")  
        files = glob.glob(images_path)
        num_images = len(files)
        print('There are {} images in {}'.format(num_images, images_path))
        df = pd.read_csv(files[0])

        images = get_image_from_csv(df)
            
        for i in range(1, num_images): 
            #print(files[i])
            df = pd.read_csv(files[i])
            res = get_image_from_csv(df)
            images = np.concatenate((images,res))
                
    if(not return_np_array):
        return images
    
    print(images.shape)
    return images

class DataReader:

    def __init__(self, train_path, eval_path, test_path, is_training=True, SHOW_IMAGES=False): 
    
        self.rotation_degrees = [0, 90, 180, 270]
        self.SHOW_IMAGES = SHOW_IMAGES        
        if is_training:
            self.train_images_in = read_all_patches_from_directory_csv(train_path, 'inputs') 
            self.train_images_gt = read_all_patches_from_directory_csv(train_path, 'ground_truth') 
            self.train_images_in, self.train_images_gt = shuffle(self.train_images_in, self.train_images_gt) 
            self.num_train_images = len(self.train_images_in)
            self.dim_patch_in_rows = self.train_images_in.shape[1] 
            self.dim_patch_in_cols = self.train_images_in.shape[2]
            self.dim_patch_gt_rows = self.train_images_gt.shape[1] 
            self.dim_patch_gt_cols = self.train_images_gt.shape[2] 
            self.index_train = 0 
            print('number of train images is %d' % self.num_train_images)

    def get_next_batch_train(self, iteration, batch_size=32):
        end = self.index_train + batch_size 
        if iteration == 0:  # because we use only full batch
            self.index_train = 0
            end = batch_size 
            self.train_images_in, self.train_images_gt = shuffle(self.train_images_in, self.train_images_gt) 
            
        input_images = np.zeros((batch_size, self.dim_patch_in_rows, self.dim_patch_in_cols, params.num_channels))
        output_images = np.zeros((batch_size, self.dim_patch_gt_rows, self.dim_patch_gt_cols, params.num_channels))
        
        start = self.index_train
        for idx in range(start, end): 
            image_in = self.train_images_in[idx].copy()   
            image_gt = self.train_images_gt[idx].copy()    
            
            # augumentation
            idx_degree = random.randint(0, len(self.rotation_degrees) - 1) 
            image_in = utils.rotate(image_in, self.rotation_degrees[idx_degree])
            image_gt = utils.rotate(image_gt, self.rotation_degrees[idx_degree])
            input_images[idx - start] = image_in.copy()
            output_images[idx - start] = image_gt.copy()
                
            if self.SHOW_IMAGES:
                cv.imshow('input', input_images[idx - start]/255)
                cv.imshow('output', output_images[idx - start]/255)
                cv.waitKey(1000)
        
        self.index_train = end
        return input_images, output_images
 


class DataReader_old:

    def __init__(self, train_path, eval_path, test_path, is_training=True, SHOW_IMAGES=False): 
    
        self.rotation_degrees = [0, 180]
        self.SHOW_IMAGES = SHOW_IMAGES        
        if is_training:
            self.train_images_in = np.concatenate((utils.read_all_patches_from_directory(train_path, 'input%s' % params.dim_patch), utils.read_all_patches_from_directory(train_path, 'input%s' % params.dim_patch_2))) 
            print(self.train_images_in.shape)  
            self.train_images_gt = np.concatenate((utils.read_all_patches_from_directory(train_path, 'gt%s' % params.dim_patch), utils.read_all_patches_from_directory(train_path, 'gt%s' % params.dim_patch_2))) 
            print(self.train_images_gt.shape)
            self.train_images_in, self.train_images_gt = shuffle(self.train_images_in, self.train_images_gt) 
            self.num_train_images = len(self.train_images_in)
            self.dim_patch_in_rows = self.train_images_in.shape[1] 
            self.dim_patch_in_cols = self.train_images_in.shape[2]
            self.dim_patch_gt_rows = self.train_images_gt.shape[1] 
            self.dim_patch_gt_cols = self.train_images_gt.shape[2] 
            self.index_train = 0 
            print('number of train images is %d' % (self.num_train_images))   

    def get_next_batch_train(self, iteration, batch_size=32):
    
        end = self.index_train + batch_size 
        if iteration == 0: # because we use only full batch
            self.index_train = 0
            end = batch_size 
            self.train_images_in, self.train_images_gt = shuffle(self.train_images_in, self.train_images_gt) 
            
        input_images = np.zeros((batch_size, self.dim_patch_in_rows, self.dim_patch_in_cols, params.num_channels))
        output_images = np.zeros((batch_size, self.dim_patch_gt_rows, self.dim_patch_gt_cols, params.num_channels))
        
        start = self.index_train
        for idx in range(start, end): 
            image_in = self.train_images_in[idx].copy()   
            image_gt = self.train_images_gt[idx].copy()    
            # augumentation
            idx_degree = random.randint(0, len(self.rotation_degrees) - 1) 
            image_in = utils.rotate(image_in, self.rotation_degrees[idx_degree])
            image_gt = utils.rotate(image_gt, self.rotation_degrees[idx_degree])
            input_images[idx - start] = image_in.copy()
            output_images[idx - start] = image_gt.copy()
                
            if self.SHOW_IMAGES:
                cv.imshow('input', input_images[idx - start]/255)
                cv.imshow('output', output_images[idx - start]/255)
                cv.waitKey(1000)
        
        self.index_train = end
        return input_images, output_images
 
        