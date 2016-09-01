import scipy.ndimage.filters as filter
import numpy as np
import math
import scipy

import InterfacePreprocessing as IntPre

kernels = None

def normalize(matrix):
    sum = np.sum(matrix)
    if sum > 0.:
        return matrix / sum
    else:
        return matrix

def fonctGabor(x,y,t,f,s):
    cos = math.cos(t)
    sin = math.sin(t)

    yangle = x * cos + y * sin
    xangle = -x * sin + y * cos

    ysigma = (s-3)**2

    xsigma = s**2

    return math.exp(-((xangle**2)/xsigma+(yangle**2)/ysigma)/2.0) * math.cos(2*math.pi*f*xangle)

def genGabor(r, t, f,s ):
    final = []
    for i in xrange(r*2):
        temp = []
        for j in xrange(r*2):
            temp.append(fonctGabor(i-r, j-r, t, f, s))
        final.append(temp)
    return final

def generateKernel(nbSteps):
    liste = []
    
    steps = math.pi / nbSteps
    for i in xrange(nbSteps):
        angle = (math.pi / -2.) + (steps * i)
        kern = np.zeros([60, 60])
        for e in range(6, 9):
            for f in range(9, 18):
                kern += np.array(genGabor(30, (steps * i) + (steps / 2.), math.sqrt(2)/f, e))
        liste.append((normalize(kern), angle+steps, angle))

    global kernels
    kernels = liste

def enhacementGabor(image):
    image = image.astype(np.float64)
    newimage = np.zeros(image.shape,dtype=np.float64)

    # Create the orientation map to guide gabor
    OrMap = IntPre.orientation(image, 20)

    # Appy kernels
    global kernels
    for kernel, sup, inf in kernels:
        temp = scipy.signal.fftconvolve(image,kernel,mode='same')
        newimage += temp * np.where((OrMap<=sup) & (OrMap>=inf), 1, 0)

    # Binarisation
    newimage = np.where(newimage<np.average(newimage), 0, 1)

    # Detect where are the fingerprint
    VarMap = IntPre.variance(image, 10,  moving=True)
    VarMap = np.where(VarMap<100, 0, 1)

    # 
    newimage *= VarMap
    newimage += (VarMap-1)*-1
    
    return newimage

# If you lauch this file alone :
if __name__ == '__main__':
    print "Laugtching as Script"
    import MultiTools as Tool
    import matplotlib.image as mpimg

    # Load an image
    image = mpimg.imread('Tests/portrait.jpg')[:,:,0]

    generateKernel(16)
    
    Tool.showImg(enhacementGabor(image))
