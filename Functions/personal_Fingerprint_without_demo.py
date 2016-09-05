# coding: utf-8
import os, random
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from scipy.ndimage import imread
from scipy.misc import imresize

from sklearn.metrics import roc_curve, auc


def show_ROC(actual, predictions):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    
    plt.title('No OrMap')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
    label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


import InterfacePreprocessing as IntPre

# Create the same image merge with maximum 3 kind of pre-processing
def preprocessing(image, size, fast = True, freq = False, orient = True, varian = True):
    image_temp = imresize(image, [size//2, size//2])
    mid = size-size//2
    image_temp = np.pad(image_temp, ((0,mid), (0,mid)), mode='maximum')
    
    if fast:
        image = imresize(image, [image.shape[0]//2, image.shape[1]//2]).astype(np.float32)
    
    if freq:
        image_temp[size//2:,:size//2] = imresize(IntPre.frequency(image, step=5), [mid, size//2])
    if orient:
        image_temp[:size//2, size//2:] = imresize(np.abs(IntPre.orientation(image, coherence=False)), [size//2, mid])
    if varian:
        image_temp[size//2:,size//2:] = imresize(IntPre.variance(image), [size//2, size//2])
    
    return image_temp
    

# Functions to load images. The only rules are to have the number of the sample
# between 0 and 9 at the end of the name (ID_NAME-2.png etc...)

# To load images like CASIA (no matching etc...)
def load_vrac(path):
    images = dict()

    for f in os.listdir(path):
        ID = f[:-4]
        image = imread(path+f)
        image_temp = image
        #image_temp = preprocessing(image, image_size, freq = True)
        #image_temp = IntPre.orientation(imresize(image, [image_size*2, image_size*2]))
        images[ID] = imresize(image_temp, [image_size, image_size])

    return images

# To load images organised by pairs (of match and mismatch)
def load_pairs(path_match, path_mismatch):
    return load_vrac(path_match), load_vrac(path_mismatch)


p_matching = 0.5
image_size = 72
num_filter_1 = 32
num_filter_2 = 64

print "Loading"

# images = load_vrac('/home/cedric/Bureau/Sample/')

images_match, images_mismatch = load_pairs("./PAIRS/MATCHED/CROP/", "./PAIRS/MISMATCHED/CROP/")
    
print "Done"



# Functions used to generate the batch.

# If images are loaded with "load_vrac"
def generate_batch_vrac(images, num, image_size, nb_img = 4):
    x = np.zeros((num, image_size*image_size))
    x_p = np.zeros((num, image_size*image_size))
    y = np.zeros((num, 2))
    no_match = int(np.round(num*0.5))
    match = num - no_match
    
    keys = images.keys()
    
    for i in range(no_match):
        ID1 = random.choice(keys)
        x[i,:] = np.reshape(images[ID1], (1,image_size*image_size))
        
        ID2 = random.choice(keys)
        while ID2[:-1] == ID1[:-1]:
            ID2 = random.choice(keys)
        x_p[i,:] = np.reshape(images[ID2], (1,image_size*image_size))
        
        y[i, 0] = 1
        
    for i in range(match):
        ID1 = random.choice(keys)
        x[no_match+i,:] = np.reshape(images[ID1], (1,image_size*image_size))
        
        ID2 = ID1[:-1] + str(random.randint(0, nb_img))
        while ID2 == ID1:
            ID2 = ID1[:-1] + str(random.randint(0, nb_img))
        x_p[no_match+i,:] = np.reshape(images[ID2], (1,image_size*image_size))
        
        y[no_match+i,1] = 1
    
    return [x, x_p, y]

# If images are loaded with "load_pairs"
def generate_batch_pairs(images_match, images_mismatch, num, image_size):
    x = np.zeros((num, image_size*image_size))
    x_p = np.zeros((num, image_size*image_size))
    y = np.zeros((num, 2))
    mis_match = int(np.round(num*0.5))
    match = num - mis_match
    
    keys_match = images_match.keys()
    for i in range(mis_match):
        ID = random.choice(keys_match)[:-1]
        x[i,:] = np.reshape(images_mismatch[ID+'0'], (1,image_size*image_size))
        x_p[i,:] = np.reshape(images_mismatch[ID+'1'], (1,image_size*image_size))
        y[i, 0] = 1
        
    
    keys_mismatch = images_mismatch.keys()
    for i in range(match):
        ID = random.choice(keys_mismatch)[:-1]
        x[mis_match+i,:] = np.reshape(images_match[ID+'0'], (1,image_size*image_size))
        x_p[mis_match+i,:] = np.reshape(images_match[ID+'1'], (1,image_size*image_size))
        y[mis_match+i, 1] = 1
        
    return [x, x_p, y]

# Transform the list of pair in to a the same
# list and here opposite (A,B -> A,B and B,A)
# to simul a siamese network
def simul_siamese(batch):
    x, x_p, y = batch
    num = len(x)
    
    new_x = np.zeros((num*2, len(x[0])))
    new_x_p = np.zeros((num*2, len(x_p[0])))
    new_y = np.zeros((num*2, len(y[0])))
    
    new_x[:num] = x
    new_x[num:] = x_p
    
    new_x_p[:num] = x_p
    new_x_p[num:] = x
    
    new_y[:num] = y
    new_y[num:] = y
    
    return [new_x, new_x_p, new_y]

# In[6]:

sess = tf.InteractiveSession()

# These two are input images
x = tf.placeholder(tf.float32, shape=[None, image_size*image_size])
x_p = tf.placeholder(tf.float32, shape=[None, image_size*image_size])

# y_ is just a value 0(match) or 1(no match) for the two input images
y_ = tf.placeholder(tf.float32, shape=[None, 2])

# Dropout coefficient
keep_prob = tf.placeholder(tf.float32)

sess.run(tf.initialize_all_variables())


# In[7]:

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.5, shape=shape)
    return tf.Variable(initial)

def assist_variable(shape):
    initial = tf.constant(0.6, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, stride=[1, 1, 1, 1]):
    return tf.nn.conv2d(x, W, strides=stride, padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
def sigm(x):
    return tf.sigmoid(x)

def prob(x):
    p_z_m *= p[i]*tf.pow(((1-p[i])/p[i]), x[i])
    p_z_no_m *= (1-q[i])*tf.pow((q[i]/(1-q[i])), z_i_flat[i])

def calc_prob(dz):
    p_m = 1 # P(dz|M)
    p_no_m = 1 # P(dz|^M)
    
    # First flatten it
    z_flat = tf.reshape(dz, [-1,9*num_filter_2])
    # Then after calculating the sigmoid, multiple everything for EACH pair
    p_match = tf.reduce_prod(sigm(z_flat),reduction_indices=[1])

    # P(M|dz)
    prob = p_m * p_matching / (p_m * p_matching + p_no_m * (1 - p_matching))


# # Defining Shared Weights

# In[8]:

W_conv1 = weight_variable([7, 7, 1, num_filter_1])
b_conv1 = bias_variable([num_filter_1])

W_conv2 = weight_variable([5, 5, num_filter_1, num_filter_2])
b_conv2 = bias_variable([num_filter_2])

W_conv1_p = weight_variable([7, 7, 1, num_filter_1])
b_conv1_p = bias_variable([num_filter_1])

W_conv2_p = weight_variable([5, 5, num_filter_1, num_filter_2])
b_conv2_p = bias_variable([num_filter_2])

# p = assist_variable((num_filter_2,))
# q = assist_variable((num_filter_2,))

W_fc1 = weight_variable([576, 1024])
b_fc1 = bias_variable([1024])

W_fc2 = weight_variable([1024, 256])
b_fc2 = bias_variable([256])

W_fc3 = weight_variable([256, 2])
b_fc3 = bias_variable([2])


# 

# In[9]:

# Input Image
x_image = tf.reshape(x, [-1,image_size,image_size,1])

# First Conv Layer
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, stride=[1, 3, 3, 1]) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second Conv Layer
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, stride=[1, 2, 2, 1]) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Final Data Processing Step
z = sigm(h_pool2)


# # Side 2

# In[10]:

# Input Image
x_image_p = tf.reshape(x, [-1,image_size,image_size,1])

# First Conv Layer
h_conv1_p = tf.nn.relu(conv2d(x_image_p, W_conv1_p, stride=[1, 3, 3, 1]) + b_conv1_p)
h_pool1_p = max_pool_2x2(h_conv1_p)

# Second Conv Layer
h_conv2_p = tf.nn.relu(conv2d(h_pool1_p, W_conv2_p, stride=[1, 2, 2, 1]) + b_conv2_p)
h_pool2_p = max_pool_2x2(h_conv2_p)

# Final Data Processing Step
z_p = sigm(h_pool2_p)


# # Simple Fully Connected Layer

# In[13]:

dz = z - z_p

# First flatten it
dz_flat = tf.sigmoid(tf.reshape(dz, [-1,9*num_filter_2]))

y_fc1 = tf.nn.softmax(tf.matmul(dz_flat, W_fc1) + b_fc1)

y_fc2 = tf.nn.softmax(tf.matmul(tf.sigmoid(y_fc1), W_fc2) + b_fc2)

# Dropout 

keep_prob = tf.placeholder(tf.float32)
h_fc2_drop = tf.nn.dropout(y_fc2, keep_prob)

y_readout = tf.nn.softmax(tf.matmul(tf.sigmoid(h_fc2_drop), W_fc3) + b_fc3)


# In[94]:

#cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.abs(y_ - y_readout), reduction_indices=[1]))
#cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(y_, y_readout), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_readout), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_readout,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

for i in range(2000):
    batch = generate_batch_pairs(images_match, images_mismatch, 50, image_size)
    #batch = generate_batch_vrac(images, 50, image_size)
    batch = simul_siamese(batch)
    
    
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], x_p:batch[1], y_: batch[2], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))


#    if i%100 == 0:
#        results = y_readout.eval(feed_dict={x:batch[0], x_p:batch[1], y_: batch[2], keep_prob: 1.0})[:,1]
#        show_ROC(batch[2][:,1], results)


    train_step.run(feed_dict={x:batch[0], x_p:batch[1], y_: batch[2], keep_prob: 1.0})


batch = generate_batch_pairs(images_match, images_mismatch, 200, image_size)
batch = simul_siamese(batch)

print("Accuracy %g"%accuracy.eval(feed_dict={x:batch[0], x_p:batch[1], y_: batch[2], keep_prob: 1.0}))

results = y_readout.eval(feed_dict={x:batch[0], x_p:batch[1], y_: batch[2], keep_prob: 1.0})[:,1]
show_ROC(batch[2][:,1], results)
