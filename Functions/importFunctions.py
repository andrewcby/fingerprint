import os, random
import Variance as Var
import EnhacementContrast as EnCon
import OrientationMap as OrMap
import OrFreqMapFFT as OrFrMapFFT
import RegistrationFFT as RegFFT
import RegistrationBrute as RegBru
import RegistrationSimple as RegSimp
import RegistrationCombinee as RegComb
import MultiTools as Tool
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import imread
from scipy.misc import imresize
from sklearn.metrics import roc_curve, auc

def load_pairs_from_preprocessed(match_path, mismatch_path, image_size, num_layer):
    images_match = dict()

    for ID in range(94):
        for pic in range(2):
            image_temp = np.zeros((image_size,image_size,num_layer))
            for layer in range(num_layer):
                fname = str(ID+1)+'_'+str(pic)+'_'+str(layer)+'.png'
                image_temp[:,:,layer] = imresize(imread(match_path+fname),[image_size,image_size],interp='bicubic')
            images_match[str(ID+1)+'_'+str(pic)] = image_temp
    
    images_mismatch = dict()

    for ID in range(94):
        for pic in range(2):
            image_temp = np.zeros((image_size,image_size,num_layer))
            for layer in range(num_layer):
                fname = str(ID+1)+'_'+str(pic)+'_'+str(layer)+'.png'
                image_temp[:,:,layer] = imresize(imread(mismatch_path+fname),[image_size,image_size],interp='bicubic')
            images_mismatch[str(ID+1)+'_'+str(pic)] = image_temp
            
    return images_match, images_mismatch


def generate_batch_pairs_from_preprocessed(images_match, images_mismatch, num, image_size, num_layer, once=False):
        
    x = np.zeros((num, image_size*image_size*num_layer))
    x_p = np.zeros((num, image_size*image_size*num_layer))
    y = np.ones((num, 1))*-1
    match = int(np.round(num*0.5))
    mis_match = num - match

    keys_mismatch = images_mismatch.keys()

    if mis_match > len(keys_mismatch):
        mis_match = len(keys_mismatch)
        print 'Number of Mismatch wanted to big !!!!'

    for i in range(mis_match):
        ID = random.choice(keys_mismatch)[:-1]
        img = images_mismatch[ID + '0']
        img_p = images_mismatch[ID + '1']
        # To use only one each pairs
        if once:
            keys_mismatch.remove(ID+'0')
            keys_mismatch.remove(ID+'1')

        x[i,:] = np.reshape(img, (1,image_size*image_size*num_layer))
        x_p[i,:] = np.reshape(img_p, (1,image_size*image_size*num_layer))
        y[i] = 0

    keys_match = images_match.keys()

    if match > len(keys_match):
        match = len(keys_match)
        print 'Number of Match wanted to big !!!!'

    for i in range(match):
        ID = random.choice(keys_match)[:-1]
        img = images_match[ID + '0']
        img_p = images_match[ID + '1']
        # To use only each pairs once
        if once:
            keys_match.remove(ID+'0')
            keys_match.remove(ID+'1')
        x[mis_match+i,:] = np.reshape(img, (1,image_size*image_size*num_layer))
        x_p[mis_match+i,:] = np.reshape(img_p, (1,image_size*image_size*num_layer))
        y[mis_match+i] = 1
    
    return [x, x_p, y]


def suffle_all(x, x_p, y):
    x_shuffle = []
    x_p_shuffle = []
    y_shuffle = []
    
    shuffle = range(len(x))
    random.shuffle(shuffle)
    
    for i in shuffle:
        x_shuffle.append(x[i])
        x_p_shuffle.append(x_p[i])
        y_shuffle.append(y[i])

    return [x_shuffle, x_p_shuffle, y_shuffle]

def show_ROC(actual, predictions, title):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title(title)
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.figure(figsize=(100, 100), dpi= 80, facecolor='w', edgecolor='k')
    plt.show()
