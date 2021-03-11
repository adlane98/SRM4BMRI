# Based on the work of  Mariana-Iuliana Georgescu, Radu Tudor Ionescu, Nicolae Verga
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import cv2 as cv 
import params_hw
import params_d
import pdb
import re
import os
import nibabel as nib
import matplotlib.pyplot as plt
import utils

#params_hw.show_params()

config = tf.ConfigProto(
        device_count={'GPU': 1}
    ) 


def upscale(downscaled_image, checkpoint):

    scale_factor = params_hw.scale   
     
    # cnn resize  
    input = tf.placeholder(tf.float32, (1, downscaled_image.shape[1], downscaled_image.shape[2], params_hw.num_channels), name='input')
    _, output = params_hw.network_architecture(input) 

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
        #cnn_output[cnn_output > 255] = 255 

        return cnn_output[:,:,:,0]

def run_network(downscaled_image, checkpoint):
    scale_factor = params_d.scale  
    
    # cnn resize  
    input = tf.placeholder(tf.float32, (1, downscaled_image.shape[1], downscaled_image.shape[2], params_d.num_channels), name='input')  
    _, output = params_d.network_architecture(input, is_training=False) 

    with tf.Session(config=config) as sess:  
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        print('restoring from ' + checkpoint)
        saver.restore(sess, checkpoint)
         
        # step 1 - apply cnn on each resized image, maybe as a batch 
        cnn_output = []
        for image in downscaled_image: 
            cnn_output.append(sess.run(output, feed_dict={input: [image]})[0])
    
        cnn_output = np.array(cnn_output)   
        cnn_output = np.round(cnn_output) 
        #cnn_output[cnn_output > 255] = 255 
        return cnn_output

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

def reshape(img):
    x,y,z = img.shape
    res = np.zeros((z,x,y))
    temp = np.zeros((x,z,y))
    for i in range(x):
        temp[i] = img[i].T
    for i in range(y):
        res[:,:,i] = temp[:,:,i].T
    return res


# Settings
scale = 2
dir_path =  r'D:\Utilisateurs\Alexandre\Repertoire_D\projet_super_resolution\data\Marmoset'
mri_name =  r'.\4299.nii'
img_path =  dir_path + mri_name
model_path_hw = r'.\build\model_hw_ckpt_10mri_1401\model.ckpt39'
model_path_d = r'.\build\model_d_ckpt_10mri_1401\model.ckpt39'

file_name = "marmouset_stest2_upscale_with_10mri_train.nii.gz"
optional = ""
DOWNSCALE = True

# Some optional parameter
VISUALIZATION = True
BLUR_INPUT = True
SAVE_INPUT = True
SAVE_OUTPUT_1ST_MODEL = True 

# Visualization parameter
slices_value = [10,20,30]

# Blur parameter
sigma = 1.5
kernel_size = (7,7)

img_3d = nib.load(img_path)
ground_truth_img = img_3d.get_fdata()

if DOWNSCALE:
    input_img = downscale_mri(ground_truth_img,scale)
else:
    input_img = ground_truth_img

if BLUR_INPUT:
    

    x,y,z = input_img.shape
    input_img_blr = cv.GaussianBlur(input_img[:,:,:],kernel_size,sigma)
    img = nib.Nifti1Image(input_img_blr,affine=img_3d.affine)
    nib.save(img,"4299_blur_1_5_7x7.nii.gz")
    optional = "blur"
print(optional)

input_img2 = np.swapaxes(input_img,1,2)
input_img2 = np.swapaxes(input_img2,0,1)

image_wh = upscale(input_img2,model_path_hw)
image_wh = np.swapaxes(image_wh,0,1)
image_wh = np.swapaxes(image_wh,1,2)
tf.reset_default_graph()

image_wh = np.expand_dims(image_wh, axis=3)
image_d = run_network(image_wh,model_path_d)

final_img = image_d[:,:,:,0]

if DOWNSCALE and VISUALIZATION:

    for i in slices_value:
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(input_img[i],cmap='gray', vmin=0, vmax=255)
        plt.xlabel("input")
        plt.subplot(1,3,2)
        plt.imshow(final_img[i*scale],cmap='gray', vmin=0, vmax=255)
        plt.xlabel("predicted")
        plt.subplot(1,3,3)
        plt.imshow(ground_truth_img[i*scale],cmap='gray', vmin=0, vmax=255)
        plt.xlabel("ground_truth")
        plt.show()


x, y, z = final_img.shape
ground_truth_img_adapt = ground_truth_img[:x,:y,:z]

img = nib.Nifti1Image(final_img,affine=img_3d.affine)
nib.save(img,file_name)
print(final_img.shape)

print("Save to: "+file_name)

if DOWNSCALE:
    print("PSNR:",utils.psnr(final_img,ground_truth_img_adapt))
    print("SSIM:",utils.ssim(final_img,ground_truth_img_adapt))

