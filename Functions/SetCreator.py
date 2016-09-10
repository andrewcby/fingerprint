# -*- coding: cp1252 -*-
import os
from scipy import ndimage
import random
from random import randint
import scipy
import numpy as np

import RegistrationCombinee as RegComb

import MultiTools as Tools

import InterfacePreprocessing as InterPre

ID = 0

def dont_hit_the_edge((x, y), (xmin, xmax, ymin, ymax)):
    x, y = x-1, y-1
    
    if xmin<0:
        xmax -= xmin
        xmin = 0

    if ymin<0:
        ymax -= ymin
        ymin = 0

    if xmax>x:
        xmin -= xmax - x
        xmax = x

    if ymax>y:
        ymin -= ymax - y
        ymax = y

    return (xmin, xmax, ymin, ymax)

# Apply all preprocessing to img, crop them to the good size
# and save all in path with name_0.png, name_1.png etc...
def generate_maps(img, name, path, (xmin, xmax, ymin, ymax)):
    w, h = len(img), len(img[0])
    
    path += name

#    xmin, xmax, ymin, ymax = 0, w, 0, h

    scipy.misc.toimage(img[xmin:xmax,ymin:ymax], cmin=0, cmax=255).save(path + '_0.png')

    # Crop with a margin to prevent side effect
    # and accelerate the preprossessing
    margin = 30
    if xmin>margin:
        img = img[xmin-margin:,:]
        xmax -= xmin-margin
        xmin = margin
    if w-xmax>margin:
        img = img[:xmax+margin,:]
    if ymin>margin:
        img = img[:,ymin-margin:]
        ymax -= ymin-margin
        ymin = margin
    if w-ymax>margin:
        img = img[:,:ymax+margin]

    # FreqMap are the slowest step, but they can work
    # faster with step=2 or 3
    FreqMap = InterPre.frequency(img, block=20, step=1, smoothFreq=True)[xmin:xmax,ymin:ymax]
    scipy.misc.toimage(FreqMap, cmin=np.min(FreqMap), cmax=np.max(FreqMap)).save(path + '_1.png')
    
    OrMap = InterPre.orientation(img, 30)[xmin:xmax,ymin:ymax]
    scipy.misc.toimage(OrMap, cmin=np.min(OrMap), cmax=np.max(OrMap)).save(path + '_2.png')

    OrMap = np.abs(OrMap)
    scipy.misc.toimage(OrMap, cmin=np.min(OrMap), cmax=np.max(OrMap)).save(path + '_3.png')
    
    VarMap = InterPre.variance(img, block=20)[xmin:xmax,ymin:ymax]
    scipy.misc.toimage(VarMap, cmin=np.min(VarMap), cmax=np.max(VarMap)).save(path + '_4.png')
    
    Gabor = InterPre.gabor(img, precision=20)[xmin:xmax,ymin:ymax]
    scipy.misc.toimage(Gabor, cmin=np.min(Gabor), cmax=np.max(Gabor)).save(path + '_5.png')

# Take the two images and return two
# image registred and coordinate of
# the center of the two fingerprints
def registre_images(img1, img2):
    (img1, img2), center = RegComb.registration(img1, img2, True)
    return ((img1, img2), center)

# Take the path of the two images, the path
# for the out and the final windows to :
# charge, translate, rotate and call generate_maps
# to save all that
def do_all(img_in1, img_in2, path_out, rect):
    global ID
    
    img1 = ndimage.imread(img_in1, mode = 'I')
    img2 = ndimage.imread(img_in2, mode = 'I')

    # Beggin Cropping of the image ----------------
    #
    seg_img1 = InterPre.variance(img1, 10)
    seg_img2 = InterPre.variance(img2, 10)

    xMin, xMax, yMin, yMax = InterPre.varianceCrop(seg_img1, seuil=15)
    img1 = img1[xMin:xMax,yMin:yMax]

    xMin, xMax, yMin, yMax = InterPre.varianceCrop(seg_img2, seuil=15)
    img2 = img2[xMin:xMax,yMin:yMax]
    #
    # End Cropping --------------------------------

    (img1, img2), center = registre_images(img1, img2)

    center = len(img1)//2, len(img1[0])//2

    xmin, xmax, ymin, ymax = center[0]+rect[0], center[0]+rect[1], center[1]+rect[2], center[1]+rect[3]

    xmin, xmax, ymin, ymax = dont_hit_the_edge((len(img1), len(img1[0])), (xmin, xmax, ymin, ymax))

    generate_maps(img1, str(ID) + "_0", path_out, (xmin, xmax, ymin, ymax))
    generate_maps(img2, str(ID) + "_1", path_out, (xmin, xmax, ymin, ymax))

    # To see the supperposition
    scipy.misc.toimage(img1 + img2, cmin=0, cmax=510).save(path_out + str(ID) + '_5.png')

    ID += 1

def generate_random_path():
    num = str(randint(0,4)) + str(randint(0,4)) + str(randint(0,4))
    side = random.choice(["L","R"])
    file = num + "_" + side + str(randint(0,3)) + "_" + str(randint(0,4)) + ".bmp"

    return num + "/" + side + "/" + file

def fast_create_match_set(path_in, path_out, n, rect=(-50, 50, -50, 50)):
    
    for i in range(n):
        num = str(randint(0,4)) + str(randint(0,4)) + str(randint(0,4))
        side = random.choice(["L","R"])
        other = str(randint(0,3))

        file = path_in + num + "/" + side + "/" + num + "_" + side + other + "_"

        tmp = [str(j) for j in range(5)]
        
        path_1 = file + tmp.pop(randint(0, 4)) + ".bmp"

        path_2 = file + tmp.pop(randint(0, 3)) + ".bmp"
        
        do_all(path_1, path_2, path_out, rect)

def fast_create_mismatch_set(path_in, path_out, n, rect=(-50, 50, -50, 50)):
    
    for i in range(n):

        path_1 = path_in + generate_random_path()

        path_2 = path_1

        while path_1[:-6] == path_2[:-6]:
            path_2 = path_in + generate_random_path()
        
        do_all(path_1, path_2, path_out, rect)

fast_create_match_set("./CASIA/", "./Preprocessed_CASIA", 3, rect=(-90, 90, -90, 90))

#do_all("D:/Recherche/DBB/CASIA-FingerprintV5(BMP)/313/R/313_R2_2.bmp", "D:/Recherche/DBB/CASIA-FingerprintV5(BMP)/313/R/313_R2_0.bmp", "D:/Recherche/Programmation/SetCreator/", rect=(-90, 90, -90, 90))
