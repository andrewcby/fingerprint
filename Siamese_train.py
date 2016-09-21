
# coding: utf-8

# In[1]:

import os, random

import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt


from sys import path
from PIL import Image
from scipy.ndimage import imread
from scipy.misc import imresize
from scipy.stats import pearsonr
from sklearn.metrics import roc_curve, auc

code_dir = os.path.join(os.path.abspath("."), "Functions")
path.append(code_dir)
data_dir = './PreProcessed/'

import InterfacePreprocessing as IntPre
import importFunctions as iF

# # Defining global variables

# In[2]:

global raw_only, image_size, num_layer
raw_only = False
raw_image_size = 150
image_size= 88

if raw_only:
    num_layer = 1
else:
    num_layer = 6

p_matching = 0.5
num_filter_1 = 20
num_filter_2 = 40
num_filter_3 = 60

    

# ################## #
# Helper Func for tf #
# ################## #

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.3, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, stride=[1, 1, 1, 1]):
    return tf.nn.conv2d(x, W, strides=stride, padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
def sigm(x):
    return tf.sigmoid(x)


# In[4]:

# ################## #
#      Load Data     #
# ################## #

# Smaller Dataset
images_match_small, images_mismatch_small = iF.load_pairs_from_preprocessed("../Fingerprint_Data/PreProcessed/Match/", 
                                                                "../Fingerprint_Data/PreProcessed/MisMatch/", 
                                                                image_size, num_layer, True)

# # Full Dataset
images_match, images_mismatch = iF.load_pairs_from_preprocessed("../Fingerprint_Data/Processed_Full_CASIA/Match/", 
                                                                "../Fingerprint_Data/Processed_Full_CASIA/MisMatch/", 
                                                                image_size, num_layer, False)



# # Creating a new TF session and defining Shared Weights

# In[7]:

# These two are input images
x = tf.placeholder(tf.float32, shape=[None, image_size*image_size*num_layer])
x_p = tf.placeholder(tf.float32, shape=[None, image_size*image_size*num_layer])

# y_ is just a value 0(match) or 1(no match) for the two input images
y_ = tf.placeholder(tf.float32, shape=[None, 1])

# Dropout coefficient
keep_prob = tf.placeholder(tf.float32)

# sess.run(tf.initialize_all_variables())

W_conv1 = weight_variable([7, 7, num_layer, num_filter_1])
b_conv1 = bias_variable([num_filter_1])

W_conv2 = weight_variable([5, 5, num_filter_1, num_filter_2])
b_conv2 = bias_variable([num_filter_2])

W_conv3 = weight_variable([5, 5, num_filter_2, num_filter_3])
b_conv3 = bias_variable([num_filter_3])

W_conv1_p = weight_variable([7, 7, num_layer, num_filter_1])
b_conv1_p = bias_variable([num_filter_1])

W_conv2_p = weight_variable([5, 5, num_filter_1, num_filter_2])
b_conv2_p = bias_variable([num_filter_2])

W_conv3_p = weight_variable([5, 5, num_filter_2, num_filter_3])
b_conv3_p = bias_variable([num_filter_3])

W_fc1 = weight_variable([image_size/8*image_size/8*num_filter_3, 1024])
b_fc1 = bias_variable([1024])

W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])


# # Two Sides of Siamese Net

# In[8]:

# ############### #
#      Side 1     #
# ############### #

# Input Image
x_image = tf.reshape(x, [-1,image_size,image_size,num_layer])

# First Conv Layer - after maxpool 44*44
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second Conv Layer - after maxpool 22*22
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Third Conv Layer - after maxpool 11*11
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

# Final Data Processing Step
z = tf.reshape(h_pool3, [-1,image_size/8*image_size/8*num_filter_3])
# z = tf.reshape(h_pool3, [-1,4*4*256])
z_norm = tf.pow(tf.reduce_sum(tf.pow(z, 2), reduction_indices=1),0.5)

h_fc1 = tf.nn.relu(tf.matmul(z, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# ############### #
#      Side 2     #
# ############### #

# Input Image
x_image_p = tf.reshape(x_p, [-1,image_size,image_size,num_layer])

# First Conv Layer - after maxpool 44*44
h_conv1_p = tf.nn.relu(conv2d(x_image_p, W_conv1_p) + b_conv1_p)
h_pool1_p = max_pool_2x2(h_conv1_p)

# Second Conv Layer - after maxpool 22*22
h_conv2_p = tf.nn.relu(conv2d(h_pool1_p, W_conv2_p) + b_conv2_p)
h_pool2_p = max_pool_2x2(h_conv2_p)

# Third Conv Layer - after maxpool 11*11
h_conv3_p = tf.nn.relu(conv2d(h_pool2_p, W_conv3_p) + b_conv3_p)
h_pool3_p = max_pool_2x2(h_conv3_p)

# Final Data Processing Step
z_p = tf.reshape(h_pool3_p, [-1,image_size/8*image_size/8*num_filter_3])
# z_p = tf.reshape(h_pool3_p, [-1,4*4*256])
z_p_norm = tf.sqrt(tf.reduce_sum(tf.square(z_p), reduction_indices=1))

h_fc1_p = tf.nn.relu(tf.matmul(z_p, W_fc1) + b_fc1)
h_fc1_p_drop = tf.nn.dropout(h_fc1, keep_prob)

y_conv_p=tf.nn.softmax(tf.matmul(h_fc1_p_drop, W_fc2) + b_fc2)


# # Actual Calculation


# Cosine

abs_dist = tf.div(tf.reduce_sum(z*z_p, reduction_indices=1), z_norm*z_p_norm)
distance = tf.mul(tf.div(tf.reduce_sum(z*z_p, reduction_indices=1), z_norm*z_p_norm), tf.transpose(y_))
cross_entropy = -tf.reduce_sum(distance, reduction_indices=1)

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

auc_list = []

saver = tf.train.Saver({"W1": W_conv1, "W2": W_conv2,"W3": W_conv3,
                        "b1": b_conv1,"b2": b_conv2,"b3": b_conv3,
                        "W1_p": W_conv1_p, "W2_p": W_conv2_p,"W3_p": W_conv3_p,
                        "b1_p": b_conv1_p,"b2_p": b_conv2_p,"b3_p": b_conv3_p,
                        "W_fc1": W_fc1, "b_fc1": b_fc1,
                        "W_fc2": W_fc2, "b_fc2": b_fc2})

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    abs_dist = tf.div(tf.reduce_sum(z*z_p, reduction_indices=1), z_norm*z_p_norm)
    distance = tf.mul(tf.div(tf.reduce_sum(z*z_p, reduction_indices=1), z_norm*z_p_norm), tf.transpose(y_))
    cross_entropy = -tf.reduce_sum(distance, reduction_indices=1)
    sess.run(tf.initialize_all_variables())
    
    f = open('log.txt', 'w+')
    for i in range(10000):
        batch = iF.generate_batch_pairs_from_preprocessed(images_match, images_mismatch, 50, image_size, num_layer)
	
	if i < 100:
	    if i%10 == 0:
		print i		

        if i% 100 == 0:
            auc_batch = iF.generate_batch_pairs_from_preprocessed(images_match_small, images_mismatch_small, 50, image_size, num_layer)
            d =  abs_dist.eval(feed_dict={x:auc_batch[0], x_p:auc_batch[1], y_: auc_batch[2], keep_prob: 1.0})
            fpr, tpr, _ = roc_curve(auc_batch[2], d.T)
            roc_auc = auc(fpr, tpr)
            auc_list.append(roc_auc)
            str1 =  'Iteration '+ str(i) 
            str2 =  ' AUC: %.2f'%roc_auc+' Loss: '+str(cross_entropy.eval(feed_dict={x:auc_batch[0], x_p:auc_batch[1], y_: auc_batch[2], keep_prob: 1.0})[0])
            str3 =  ' Distance: '+str(((d[1:6]*100).astype(int)).astype(float)/100)+ ' Labels: '+ str((auc_batch[2][1:6].T).astype(int))
            save_path = saver.save(sess, "training"+str(i)+".ckpt")
            str4 = "Model saved in file: %s" % save_path
            
            
            f.write(str1 + str2+ str3 + '\n')
            f.write(str4+ '\n')
            
            f2 = open('progress.txt', 'w+')
            f2.write(str1 + str2+ str3 + '\n')
            f2.write(str4+ '\n')
            f2.close
            
        train_step.run(feed_dict={x:batch[0], x_p:batch[1], y_: batch[2], keep_prob: 1.0})
        
    f.close()
