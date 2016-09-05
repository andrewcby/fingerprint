# -*- coding: cp1252 -*-
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.fftpack import fftn, ifftn, fftshift
from scipy import ndimage
import scipy.ndimage.interpolation as ndii
import math

from PIL import Image

xLogpolar = None
yLogpolar = None

def pre_logpolar(shape):
    # Source : http://www.lfd.uci.edu/~gohlke/code/imreg.py.html
    center = shape[0] / 2, shape[1] / 2
    angles = shape[0]
    radii = shape[1]
    theta = np.empty(shape, dtype=np.float64)
    theta.T[:] = -np.linspace(0, np.pi, shape[0], endpoint=False)
    d = np.hypot(shape[0]-center[0], shape[1]-center[1])
    log_base = 10.0 ** (math.log10(d) / (radii))
    radius = np.empty_like(theta)
    radius[:] = np.power(log_base, np.arange(radii, dtype=np.float64)) - 1.0
    global xLogpolar
    global yLogpolar
    xLogpolar = radius * np.sin(theta) + center[0]
    yLogpolar = radius * np.cos(theta) + center[1]

def logpolar(img):
    global xLogpolar
    global yLogpolar
    output = np.empty_like(xLogpolar)
    ndii.map_coordinates(img, [xLogpolar, yLogpolar], output=output)
    return output

# https://en.wikipedia.org/wiki/Phase_correlation#Method
def calc_phase(FFT1, FFT2):
    temp = abs(ifftn((FFT1 * FFT2.conjugate()) / (abs(FFT1) * abs(FFT2))))
    return np.unravel_index(np.argmax(temp), temp.shape)

def registration(img1, img2, value = False):
    height, width = img1.shape

    # Save it because re-use after
    FFT1 = fftn(img1)
    
    t1 = fftshift(abs(FFT1))
    t2 = fftshift(abs(fftn(img2)))

    # Change cartesian space to logpolar
    # (to find easly rotation (and zoom)
    t1 = logpolar(t1)
    t2 = logpolar(t2)

    angle, i1 = calc_phase(fftn(t1), fftn(t2))
    
    angle = 180.0 * angle / height
    if angle < -90.0:
        angle += 180.0
    elif angle > 90.0:
        angle -= 180.0

    if 2>angle>-2 or angle>30 or angle<-35:
        angle = 0

    img2_r = ndii.rotate(img2, angle, order = 1, mode = 'reflect', reshape = False, cval = np.max(img2))

    # Now find translation
    x, y = calc_phase(FFT1, fftn(img2_r))
    
    if x > height // 2:
        x -= height
    if y > width // 2:
        y -= width

    if value:
        return angle, x, y
    
    img1 = img1[max(0, x):min(x, 0)+height,max(0, y):min(y, 0)+width]
    img2_r = ndii.rotate(img2, angle, order = 1, reshape = False, cval = np.max(img2))
    img2 = img2_r[max(-x, 0):min(-x, 0)+height, max(-y, 0):min(-y, 0)+width]

    return img1, img2


# If you lauch this file alone :
if __name__ == '__main__':
    print "Laugtching as Script"

    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex='col', sharey='row')

    ########## Calcul images ##########
    #  Example of code that you will  #
    #  write if you import this file  #

    # Importation of images as array
    image1 = ndimage.imread('D:\\Recherche\\DBB\\PERSO\\2_0.png')
    image2 = ndimage.imread('D:\\Recherche\\DBB\\PERSO\\2_4.png')

    # Generate a matrice of transformation (logpolar)
    # allays the same if the shape is the same
    # so we can generat it one for all
    pre_logpolar(image1.shape)

    # Apply registration between the two images
    image1_final, image2_final = registration(image1, image2)

    #                                 #
    #                                 #
    ########## Show results ###########
    #                                 #
    #                                 #
    ax3.imshow(image1_final,cmap=plt.cm.gray)
    ax6.imshow(image2_final,cmap=plt.cm.gray)

    ax5.imshow(image1_final + image2_final,cmap=plt.cm.gray)

    ax1.imshow(image1,cmap=plt.cm.gray)
    ax4.imshow(image2,cmap=plt.cm.gray)

    plt.show()

