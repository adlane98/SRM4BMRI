#  Copyright (c) 2020. Mariana-Iuliana Georgescu, Radu Tudor Ionescu, Nicolae Verga
#  Convolutional Neural Networks with  Intermediate Loss for 3D Super-Resolution of CT and MRI Scans
#  Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
#  (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#
# Changed by Alexandre FICHE and Maxime FURET 

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import cv2 as cv
import random
import math
from sklearn.utils import shuffle
import pdb
import re
import data_reader as reader

import networks as nets
import utils
import params

# Have a directory for patch with this tree
# -train_data|-3T 
#            |-inputs 
#            |-ground_truth   

SHOW_IMAGES = False 
IS_RESTORE = tf.train.latest_checkpoint(params.folder_data) is not None

params.show_params()   
# CHANGE YOUR PATH HERE 
path = r'D:\Utilisateurs\Alexandre\Repertoire_D\projet_super_resolution\data\marmoset_train_2'

data_reader = reader.DataReader(path, './data/', './data/')

# training 
batch_size = 128
stride = 0
input = tf.placeholder(tf.float32, (batch_size, data_reader.dim_patch_in,  data_reader.dim_patch_in, params.num_channels), name='input')
target = tf.placeholder(tf.float32, (batch_size, data_reader.dim_patch_gt - 2*stride, data_reader.dim_patch_gt - 2*stride, params.num_channels), name='target')

output_PS, output = params.network_architecture(input)  
print('output shape is ', output.shape, target.shape) 

if params.LOSS == params.L1_LOSS:
	loss = tf.reduce_mean(tf.reduce_mean(tf.abs(output - target)) + tf.reduce_mean(tf.abs(output_PS - target)))
if params.LOSS == params.L2_LOSS:
	loss = tf.reduce_mean(tf.reduce_mean(tf.square(output - target)) + tf.reduce_mean(tf.square(output_PS - target))) 	
	 
global_step = tf.Variable(0, trainable=False)
lr = params.learning_rate 
starter_learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate_epoch") 
 
 
opt = tf.train.AdamOptimizer(starter_learning_rate).minimize(loss, global_step=global_step)

config = tf.ConfigProto(
        device_count={'GPU': 1}
    ) 
config.gpu_options.allow_growth=True
   
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer()) 

total_loss_placeholder = tf.placeholder(tf.float32, shape=[], name="total_loss")
lr_placeholder = tf.placeholder(tf.float32, shape=[], name="total_loss")
ssim_placeholder = tf.placeholder(tf.float32, shape=[], name="ssim_placeholder")
psnr_placeholder = tf.placeholder(tf.float32, shape=[], name="psnr_placeholder")
tf.summary.scalar('loss', total_loss_placeholder) 
tf.summary.scalar('learning_rate', lr_placeholder)  
tf.summary.scalar('ssim', ssim_placeholder)  
tf.summary.scalar('psnr', psnr_placeholder)  
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('train.log', sess.graph)
  
saver = tf.train.Saver(max_to_keep=0)   
start_epoch = 0

if IS_RESTORE:
    print('===========restoring from ' + tf.train.latest_checkpoint(params.folder_data))
    saver.restore(sess,tf.train.latest_checkpoint(params.folder_data))
    start_epoch = re.findall(r'\d+', tf.train.latest_checkpoint(params.folder_data))
    start_epoch = int(start_epoch[0]) + 1
else:
	print("===========NEW MODEL")
#/!\ It can have an issue when restore the model, so we have to manually set the start_epoch
#start_epoch = 20

print('the number of images is: ', data_reader.num_train_images)
print("we start at ",start_epoch," epoch")
print("the goal is ",params.num_epochs, "epoch")

# Some log to display after the training
list_psnr = []
list_ssim = []
list_loss = []

for epoch in range(start_epoch, params.num_epochs):
	batch_loss = 0
	num_images = 0
	num_iterations = math.floor(data_reader.num_train_images / batch_size)
	print('the number of iterations is %d' % num_iterations)
	ssim_epoch = 0
	psnr_epoch = 0
	for i in range(0, num_iterations): 
		input_, target_  = data_reader.get_next_batch_train(i, batch_size)
		num_images += batch_size

		cost, _, predicted_images = sess.run([loss, opt, output], feed_dict={input: input_, target: target_,
																			 starter_learning_rate: lr})
		batch_loss += cost * batch_size
		ssim_batch, psnr_batch = utils.compute_ssim_psnr_batch(np.round(predicted_images), np.round(target_))
		ssim_epoch += ssim_batch
		psnr_epoch += psnr_batch
	print("Epoch/Iteration {}/{} ...".format(epoch, i), "Training loss: {:.4f}  ssim: {:.4f} psnr: {:.4f}".
			format(batch_loss/num_images, ssim_epoch/num_images, psnr_epoch/num_images), "Learning rate:  {:.8f}".
			format(lr))
	list_loss += [batch_loss/num_images]
	list_ssim += [ssim_epoch/num_images]
	list_psnr += [psnr_epoch/num_images]
	merged_ = sess.run(merged, feed_dict={total_loss_placeholder: batch_loss/num_images,
										  ssim_placeholder: ssim_epoch/num_images,
										  psnr_placeholder: psnr_epoch/num_images,
										  lr_placeholder : lr } )
	writer.add_summary(merged_, epoch)
	print('saving checkpoint...')
	saver.save(sess, params.folder_data + params.ckpt_name + str(epoch))

print(list_loss)
print(list_psnr)
print(list_ssim)

sess.close()