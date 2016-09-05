import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import ndimage
import math

import MultiTools as Tool

def generate(img, block=10, coherence=True):
    
    sobelX = ndimage.sobel(img, axis = 0)
    sobelY = ndimage.sobel(img, axis = 1)

    x, y = img.shape
    x, y = x//block, y//block
            
    orientMap = np.empty((x, y), dtype = float)
    coherMap = np.empty((x, y), dtype = float)

    # It's faster to pre-compute it
    sobelMult = np.multiply(sobelX, sobelY)
    sobelXPow = np.multiply(sobelX, sobelX)
    sobelYPow = np.multiply(sobelY, sobelY)

    for i in range(x):
        for j in range(y):
            i, j = i*block, j*block

            Gxy = ndimage.sum(sobelMult[i:i+block,j:j+block])
            Gxx = ndimage.sum(sobelXPow[i:i+block,j:j+block])
            Gyy = ndimage.sum(sobelYPow[i:i+block,j:j+block])
            
            i, j = i//block, j//block
            
            orientMap[i][j] = 0.5 * math.atan2(2.0*Gxy, Gxx - Gyy)

            if coherence:
                temp = Gxx+Gyy
                if temp == 0:
                    temp = 1
                coherMap[i][j] = math.sqrt(math.pow(Gxx-Gyy,2)+4.0*math.pow(Gxy, 2))/temp

    if coherence:
        return (orientMap, coherMap)
    else:
        return orientMap


def generate_glissant(img, block=5, coherence=True):
    
    sobelX = ndimage.sobel(img, axis = 0)
    sobelY = ndimage.sobel(img, axis = 1)

    # It's faster to pre-compute it
    sobelMult = np.multiply(sobelX, sobelY)
    sobelXPow = np.multiply(sobelX, sobelX)
    sobelYPow = np.multiply(sobelY, sobelY)

    arraySum = np.ones((block, block))

    Gxy = ndimage.filters.convolve(sobelMult, arraySum)
    Gxx = ndimage.filters.convolve(sobelXPow, arraySum)
    Gyy = ndimage.filters.convolve(sobelYPow, arraySum)
    
    Gxx_less_Gyy = Gxx-Gyy

    orientMap = 0.5 * np.arctan2(2.0*Gxy, Gxx_less_Gyy)

    if coherence:
        temp = Gxx+Gyy
        coherMap = np.empty(img.shape, dtype = float)
        
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                t = temp[i][j]
                if t == 0:
                    t = 1
                coherMap[i][j] = math.sqrt(math.pow(Gxx_less_Gyy[i][j],2)+4.0*math.pow(Gxy[i][j], 2))/t
        return (orientMap, coherMap)
    else:
        return orientMap

# If you lauch this file alone :
if __name__ == '__main__':
    print "Laugtching as Script"
    
    f, ((ax1, ax2, ax3)) = plt.subplots(1, 3, sharex='col', sharey='row')

    image1 = ndimage.imread('D:\\Recherche\\DBB\\FVC2002\\DB2_B\\102_1.tif', mode = 'I')

    ax1.imshow(image1,cmap=plt.cm.gray)

    orientMap = generate(image1, 20, False)

    orientMap = ndimage.zoom(orientMap, 20, order=0)

    ax2.imshow(orientMap,cmap=plt.cm.hsv)

    ax3.imshow(Tool.smoothingAngle(generate_glissant(image1, 20, False), 2),cmap=plt.cm.hsv)

    plt.show()
