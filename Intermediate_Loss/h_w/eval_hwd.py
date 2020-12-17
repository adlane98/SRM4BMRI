#  Copyright (c) 2020. Mariana-Iuliana Georgescu, Radu Tudor Ionescu, Nicolae Verga
#  Convolutional Neural Networks with  Intermediate Loss for 3D Super-Resolution of CT and MRI Scans
#  Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
#  (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import cv2 as cv 
import params
import utils
import pdb
import re
import os
import nibabel as nib
import matplotlib.pyplot as plt

params.show_params()

config = tf.ConfigProto(
        device_count={'GPU': 1}
    ) 


def upscale(downscaled_image, checkpoint):

    scale_factor = params.scale   
     
    # cnn resize  
    input = tf.placeholder(tf.float32, (1, downscaled_image.shape[1], downscaled_image.shape[2], params.num_channels), name='input')
    _, output = params.network_architecture(input) 

    with tf.Session(config=config) as sess:  
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        print('restoring from ' + checkpoint)
        saver.restore(sess, checkpoint)
         
        # step 1 - apply cnn on each resized image, maybe as a batch 
        cnn_output = []
        for image in downscaled_image: 
            cnn_output.append(sess.run(output, feed_dict={input: [image[:,:,None]]})[0])
    
        cnn_output = np.array(cnn_output)   
        cnn_output = np.round(cnn_output) 
        cnn_output[cnn_output > 255] = 255 

        return cnn_output[:,:,:,0]
        
        
def predict(downscaled_image, original_image, checkpoint):
 
    downscaled_image = upscale(downscaled_image, checkpoint)
    tf.reset_default_graph()

    ssim_cnn, psnr_cnn = utils.compute_ssim_psnr_batch(downscaled_image, original_image) 

    return ssim_cnn, psnr_cnn
    
        
def read_images(test_path):

    test_images_gt = utils.read_all_directory_images_from_directory_test(test_path, add_to_path='original')
    test_images = utils.read_all_directory_images_from_directory_test(test_path, add_to_path='input_x%d' % scale) 
    return test_images_gt, test_images


def compute_performance_indices(test_path, test_images_gt, test_images, checkpoint, write_to_summary=True):

    num_images = 0 
    ssim_cnn_sum = 0; psnr_cnn_sum = 0; ssim_standard_sum = 0; psnr_standard_sum = 0;  
    print("compute")
    for index in range(len(test_images)):
        # pdb.set_trace()
        ssim_cnn, psnr_cnn = predict(test_images[index], test_images_gt[index], checkpoint)
        tf.reset_default_graph()
        ssim_cnn_sum += ssim_cnn; psnr_cnn_sum += psnr_cnn  
        num_images += test_images[index].shape[0]
      
    print('cnn {} --- psnr = {} ssim = {}'.format(test_path, psnr_cnn_sum/num_images, ssim_cnn_sum/num_images))
    print("nani")
    if test_path.find('test') != -1 and write_to_summary is True:
     
        tf.summary.scalar('psnr_cnn', psnr_cnn_sum/num_images)   
        tf.summary.scalar('ssim_cnn', ssim_cnn_sum/num_images)  
        merged = tf.summary.merge_all() 
        writer = tf.summary.FileWriter('test.log')  
        epoch = re.findall(r'\d+', checkpoint)
        epoch = int(epoch[0]) 
        with tf.Session(config=config) as sess:
            merged_ = sess.run(merged)
            writer.add_summary(merged_, epoch)


# test_path = './../../../../images-testing/t1w' 
# eval_path = './data/train'
# scale = 2

# test_images_gt, test_images = read_images(test_path)  
# # checkpoint = tf.train.latest_checkpoint(params.folder_data)  
# # checkpoint = os.path.join(params.folder_data, 'model.ckpt%d' % 15)
# # compute_performance_indeces(test_path, test_images_gt, test_images, checkpoint, write_to_summary=False) 
# # exit()

# for i in range(30, 40):
#     checkpoint = os.path.join(params.folder_data, 'model.ckpt%d' % i)
#     compute_performance_indices(test_path, test_images_gt, test_images, checkpoint)
#     # compute_performance_indeces(eval_path, eval_images_gt, eval_images, checkpoint)  


# TEST FOR OUR PROJECT

# function that create list of gt slices and input slices
def downscale_3d_image(img, scale_factor):
    x,y,z = img.shape
    gt = np.zeros((1,z,x,y))
    inp = np.zeros((1,z,x//scale_factor,y//scale_factor))
    for i in range(z):
        gt[0,i,:,:] = img[:,:,i]
        inp[0,i,:,:] = cv.resize(img[:,:,i],(img.shape[1]//scale_factor,img.shape[0]//scale_factor))
    return gt, inp


def downscale_mri(img,scale):
    x,y,z = img.shape
    temp = np.zeros((x//scale,y//scale,z))
    res = np.zeros((x//scale,y//scale,z//scale))
    for i in range(z):
        temp[:,:,i] = cv.resize(img[:,:,i],(y//scale,x//scale))
    for i in range(x//scale):
        res[i,:,:] = cv.resize(temp[i,:,:],(z//scale,y//scale))
    return res

test_path = r'D:\Utilisateurs\Alexandre\Repertoire_D\projet_super_resolution\data\train\train_data\3T'
img_path = test_path+r'\Landman_3253_20110818_366806254_301_WIP_MPRAGE_SENSE_MPRAGE_SENSE.nii.gz'
model_path = r'.\data_ckpt\model.ckpt39'
scale = 2

test_path =  r'D:\Utilisateurs\Alexandre\Repertoire_D\projet_super_resolution\data\marmoset_train\train_data\3T'
img_path = test_path+r'\1010.nii'
model_path = r'.\data_ckpt_15122\model.ckpt39'
model_path2 = r'.\data_ckpt_d15122\model.ckpt39'

img_3d = nib.load(img_path)
data = img_3d.get_fdata()

test_images_gt, test_images = downscale_3d_image(data, scale)

# print(test_images_gt.shape)
# print(test_images.shape)

# import matplotlib.pyplot as plt
# plt.imshow(test_images[0])
# plt.show()
print(len(test_images))

#compute_performance_indices(test_path,test_images_gt,test_images,model_path,write_to_summary=False)

test_images = data
downscaled_image = upscale(data, model_path)
tf.reset_default_graph()

i = 50
plt.figure()
print(test_images.shape)
plt.imshow(test_images[i],cmap='gray_r', vmin=0, vmax=255)
plt.figure()
print(downscaled_image.shape)
plt.imshow(downscaled_image[i],cmap='gray_r', vmin=0, vmax=255)

#print(test_images_gt[0].shape)
#plt.imshow(test_images_gt[0][i],cmap='gray_r', vmin=0, vmax=255)
plt.show()

downscaled_image = upscale(downscaled_image, model_path2)
img = nib.Nifti1Image(downscaled_image,None)
nib.save(img,'test2.nii.gz')