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

test_path = r'D:\Utilisateurs\Alexandre\Repertoire_D\projet_super_resolution\data\train\train_data\3T'
img_path = test_path+r'\Landman_3253_20110818_366806254_301_WIP_MPRAGE_SENSE_MPRAGE_SENSE.nii.gz'
model_path = r'.\data_ckpt\model.ckpt39'
scale = 2

test_path =  r'D:\Utilisateurs\Alexandre\Repertoire_D\projet_super_resolution\data\Marmoset'
img_path = test_path+r'\3935.nii'
model_path = r'.\data_ckpt_15122\model.ckpt39'
model_path2 = r'.\data_ckpt_d1712\model.ckpt39'

img_3d = nib.load(img_path)
data = img_3d.get_fdata()

# Visualiaze luminosity
#flat = data.flatten()
#print(flat.shape)
#plt.figure()
#plt.hist(flat,100)
#plt.show()

test_images_gt, test_images = downscale_3d_image(data, scale)

# print(test_images_gt.shape)
# print(test_images.shape)

# import matplotlib.pyplot as plt
# plt.imshow(test_images[0])
# plt.show()
print(len(test_images))
test_images = data
data2 = np.expand_dims(data, axis=3)
print(data.shape)
print(data2.shape)
#compute_performance_indices(test_path,test_images_gt,test_images,model_path,write_to_summary=False)

image = np.expand_dims(data, axis=3)
image_d = run_network(image,model_path2)
tf.reset_default_graph()

test = reshape(image_d[:,:,:,0])
image_final = upscale(test, model_path)
tf.reset_default_graph()



i = 60

image_final = np.swapaxes(image_final,2,1)
image_final = np.swapaxes(image_final,2,0)

plt.figure()
print(data.shape)
plt.imshow(data[:,:,i],cmap='gray_r', vmin=0, vmax=255)

plt.figure()
print(image_final.shape)
plt.imshow(image_final[:,:,i*2],cmap='gray_r', vmin=0, vmax=255)


print("SUCCES")
print(data.shape," to ",image_final.shape)
plt.show()

# Visualiaze luminosity
#flat = image_final.flatten()
#print(flat.shape)
#plt.figure()
#plt.hist(flat,100)
#plt.show()

img = nib.Nifti1Image(image_final,None)
file_name = "marmouset_sr3937.nii.gz"
print("Save to: "+file_name)
nib.save(img,file_name)