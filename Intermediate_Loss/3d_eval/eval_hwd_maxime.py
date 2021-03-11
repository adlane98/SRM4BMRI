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


# downscale a 3D image by a scale factor
# params :
#     img : image that you want to downscale
#     scale : the scale factor of the down sampling
# output : 
#     the downsampled image
def downscale_mri(img,scale):
    x,y,z = img.shape
    temp = np.zeros((x//scale,y//scale,z))
    res = np.zeros((x//scale,y//scale,z//scale))
    for i in range(z):
        temp[:,:,i] = cv.resize(img[:,:,i],(y//scale,x//scale))
    for i in range(x//scale):
        res[i,:,:] = cv.resize(temp[i,:,:],(z//scale,y//scale))
    return res


# **************************************************
#                      MAIN

scale = 2

test_path =  r'D:\Utilisateurs\Alexandre\Repertoire_D\projet_super_resolution\data\Marmoset'
img_path = test_path + r'\3935.nii'
model_path = r'.\build\model_hw_ckpt_10mri_1401\model.ckpt39'
model_path2 = r'.\build\model_d_ckpt_10mri_1401\model.ckpt39'

img_3d = nib.load(img_path)
ground_truth_img = img_3d.get_fdata()


# Inference over a downscaled image
# input_img = downscale_mri(ground_truth_img,scale)
# img = nib.Nifti1Image(input_img,None)
# file_name = "marmouset_downscale_3935.nii.gz"
# print("Save to: "+file_name)
# nib.save(img,file_name)

# use only for downscale input
# input_img2 = np.swapaxes(input_img,1,2)

# use only for normal input
input_img2 = np.swapaxes(ground_truth_img,1,2)

input_img2 = np.swapaxes(input_img2,0,1)

image_wh = upscale(input_img2,model_path)
image_wh = np.swapaxes(image_wh,0,1)
image_wh = np.swapaxes(image_wh,1,2)
tf.reset_default_graph()

image_wh = np.expand_dims(image_wh, axis=3)
image_d = run_network(image_wh,model_path2)

final_img = image_d[:,:,:,0]


# plotting with downscaled input image
# i = 10
# plt.figure()
# plt.subplot(1,3,1)
# plt.imshow(input_img[i],cmap='gray', vmin=0, vmax=255)
# plt.xlabel("input")
# plt.subplot(1,3,2)
# plt.imshow(final_img[i*scale],cmap='gray', vmin=0, vmax=255)
# plt.xlabel("predicted")
# plt.subplot(1,3,3)
# plt.imshow(ground_truth_img[i*scale],cmap='gray', vmin=0, vmax=255)
# plt.xlabel("ground_truth")
# plt.show()

# plotting with normal image
# i = 30
# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(ground_truth_img[:,:,i],cmap='gray', vmin=0, vmax=255)
# plt.xlabel("input")
# plt.subplot(1,2,2)
# plt.imshow(final_img[:,:,i*scale],cmap='gray', vmin=0, vmax=255)
# plt.xlabel("predicted")
# plt.show()

# saving file
img = nib.Nifti1Image(final_img,affine=img_3d.affine)
file_name = "marmouset_sr3935_upscale_with_10mri_train_1802.nii.gz"
print("Save to: "+file_name)
nib.save(img,file_name)