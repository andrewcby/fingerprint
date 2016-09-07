from scipy import signal
import numpy as np
import math

from OrientationMap import generate_glissant as GenOr
from Variance import varianceMap_glisse as GenVar

kernels = None

# Like range but with float
# We can also use np.arange() but I see that after
def frange(nuplet):
    (x, y, jump) = nuplet
    while x < y:
        yield x
        x += jump

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

# Generate a Gabor kernel of 2r*2r
def genGabor(r, t, f, s):
    final = []
    for i in xrange(r*2):
        temp = []
        for j in xrange(r*2):
            temp.append(fonctGabor(i-r, j-r, t, f, s))
        final.append(temp)
    return final

def generateKernel(nbSteps, freq=(0.078, 0.157, 0.01), sigma=(6, 9, 1)):
    global kernels
    kernels = []

    # For each angle possible
    steps = math.pi / nbSteps
    for i in xrange(nbSteps):
        angle = (math.pi / -2.) + (steps * i)
        tmp = np.zeros([60, 60])
        # Combine some filters in one
        # to have less convolution to do
        for e in frange(sigma):
            for f in frange(freq):
                tmp += np.array(genGabor(30, (steps * i) + (steps / 2.), f, e))
        kernels.append((normalize(tmp), angle+steps, angle))

def enhacementGabor(image, extract=False):
    image = image.astype(np.float64)
    newimage = np.zeros(image.shape,dtype=np.float64)

    # Create the orientation map to guide gabor
    OrMap = GenOr(image, 20)
    
    # Appy each kernels
    global kernels
    for kernel, sup, inf in kernels:
        # FFT is faster compare to simple convolution
        tmp = signal.fftconvolve(image, kernel, mode='same')
        # Select only the part with the good orientation
        newimage += np.where((OrMap<=sup) & (OrMap>inf), tmp, 0)

    # Remove the background
    if extract:
        moyenne = np.average(newimage)
        newimage = np.where(newimage<moyenne, newimage, moyenne)
    
    return newimage

# If you lauch this file alone :
if __name__ == '__main__':
    print "Laugtching as Script"
    import MultiTools as Tool
    import matplotlib.image as mpimg

    # Load an image
    image = mpimg.imread('4.tif')[:,:]

    generateKernel(30)
    
    Tool.showImg(enhacementGabor(image))
