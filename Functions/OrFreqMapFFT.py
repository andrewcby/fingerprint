from scipy.fftpack import fftn, fftshift
from scipy import ndimage
from scipy import signal
from scipy.spatial.distance import pdist
from scipy.ndimage import filters
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math
import copy

###Creation d'une fenetre de hamming
### Source : https://mail.scipy.org/pipermail/numpy-discussion/2008-July/036110.html
##import numpy as np
##import scipy.signal as ss
##hm_len = heightmap.shape[0]
##bw2d = np.outer(signal.hamming(hm_len), np.ones(hm_len))
##bw2d = np.sqrt(bw2d * bw2d.T) # I don't know whether the sqrt is correct
##
### window the heightmap
##heightmap *= bw2d
### Ou sinon
##h = signal.hamming(n)
##ham2d = np.sqrt(np.outer(h,h))

anglMap = []

# Generate a Gaussian mask
def gkern2(kernlen=21, nsig=3):
    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen//2, kernlen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return filters.gaussian_filter(inp, nsig)

# Return the norm of a number (for show)
def norma(mat):
    mat1 = mat.real
    mat1 -= mat1.min()
    mat1 *= 255. / mat1.max()
    return mat1

# function the log norm of a number (for show)
def normalog(mat):
    mat1 = norma(mat)
    mat1 = np.log(1 + mat1)
    mat1 *= 255. / mat1.max()
    return mat1

def dec(a, b):
    a, b = float(a), float(b)
    if (a+b)==0:
        return 0.
    b *= b
    return b / (a*a + b)

# Augment the precision of the localisation
# of the maximum using the neighbour of it
def maxPrecision((x, y), img):
    height, width = img.shape

    c = img[x][y]
    xNew, yNew = x, y
    if x-1>=0 and x+1<height:
       xNew += dec(c, img[x+1][y]) - dec(c, img[x-1][y])
          
    if y-1>=0 and y+1<width:
       yNew += dec(c, img[x][y+1]) - dec(c, img[x][y-1])
    return (xNew, yNew)

# Extract information (angle, freq and proba) from the FFT image
def extractInfo(img, doAngle=True, doFrequ=True, doProb=True):

    height, width = img.shape

    tmpSum = np.sum(img * img)
    if tmpSum == 0:
        return (0., 0., 0.)

    # Algorithme proposed in "Fingerprint enhancement using STFT analysis"
    # (doi = 10.1016/j.patcog.2006.05.036), page 204
    if doAngle:
        global anglMap
        
        center = (height/2, width/2)
        
        probaAngle = {}
        
        for i in range(height):
            for j in range(width):
                
                proba = math.pow(img[i][j], 2) / tmpSum
                
                angl = anglMap[i][j]
                
                if probaAngle.has_key(angl):
                    probaAngle[angl] = probaAngle.get(angl) + img[i][j]
                else:
                    probaAngle[angl] = img[i][j]

        tempA = 0
        tempB = 0
        for key, value in probaAngle.items():
            tempA += value * math.sin(2.0 * key)
            tempB += value * math.cos(2.0 * key)

        angle = 0.5 * np.arctan2(tempA, tempB)
    else:
        angle = 0.

    a = np.unravel_index(img.argmax(), img.shape)

    # Use the intensity of the maximum to determine
    # the certainty of the estimation
    if doProb:
        proba = math.pow(img[a[0]][a[1]]*2.0, 2) / tmpSum
    else:
        proba = 0.

    if doFrequ:
        a = maxPrecision(a, img)
        b = (height//2, width//2)
    
        frequence = pdist((a, b), 'euclidean')[0]
    else:
        frequence = 0.

    return (angle, frequence, proba)

def mapping(img, block = 15, step = 2, doAngle=True, doFrequ=True, doProb=True):
    height, width = img.shape

    # Cosine windows
    bw2d = np.outer(signal.cosine(block), np.ones(block))
    # Hanning windows
    # bw2d = np.outer(signal.hanning(block), np.ones(block))

    # Transform 1D to 2D windows
    bw2d = np.sqrt(bw2d * bw2d.T)

    center = (block//2, block-block//2)

    # Create coordinate maps (use for orientationMap)
    global anglMap
    anglMap = np.empty((block, block))
    for i in range(block):
        for j in range(block):
            anglMap[i][j] = np.arctan2(i-center[0], j-center[1])

    # For the edge effect
    tmpImg = np.pad(img, (center,center), 'reflect')

    # Creation of the three new images
    result = []
    for i in range(0, height, step):
        temp = []
        for j in range(0, width, step):
            piece = tmpImg[i:i+block,j:j+block]

            # Apply the window
            piece = np.multiply(piece - np.average(piece), bw2d)

            # FFT
            imgFFT = fftn(piece)
            imgFFT = np.abs(imgFFT)
            
            imgFFT = fftshift(imgFFT)

            # Extract imformations (or, freq, prob)
            temp.append(extractInfo(imgFFT, doAngle, doFrequ, doProb))
        result.append(temp)

    return result

def smoothingFreq(maps, block = 10, sigma = 1.5):
    center = (block//2, block-block//2)

    tmpMaps = np.pad(maps, (center, center, (0,0)), 'reflect')
    proba = tmpMaps[:,:,2]
    carte = tmpMaps[:,:,1]
    
    carteProb = carte * proba

    fini = np.empty((maps.shape[0], maps.shape[1]))

    gauss = gkern2(block, sigma)

    for i in range(len(fini)):
        for j in range(len(fini[0])):

            tmpProba = np.multiply(proba[i:i+block,j:j+block], gauss)
            tmpcarteProb = np.multiply(carteProb[i:i+block,j:j+block], gauss)

            tmpSum = np.sum(tmpProba)
            if tmpSum==0:
                fini[i][j] = 0.
            else:
                fini[i][j] = np.sum(tmpcarteProb) / np.sum(tmpProba)

    return fini

# If you lauch this file alone :
if __name__ == '__main__':
    #image1 = ndimage.imread('Sans titre.png', mode = 'L')
    image1 = ndimage.imread('chikkerur2007.png', mode = 'L')
    #image1 = ndimage.imread('D:\\Recherche\\DBB\\FVC2002\\DB2_B\\108_3.tif', mode = 'L')
    #image1 = ndimage.imread('D:\\Recherche\\DBB\\Pack Similaire\\u01_o_fc_li_01.bmp', mode = 'L')
    #image1 = ndimage.imread('./Tests/FFT_4_36.png', mode = 'L')
    #image1 = ndimage.imread('./Tests/test3.jpg', mode = 'L')
    #image1 = ndimage.imread('./Tests/FFT_6_30.png', mode = 'L')
    #image1 = ndimage.imread('D:\\Recherche\\DBB\\FVC2002\\DB2_B\\102_1.tif', mode = 'L')

    imgFFT = fftn(image1)
    imgFFT_Abs = abs(imgFFT)
    imgFinal = fftshift(imgFFT_Abs)

    fig = plt.figure()

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    plt.title('Image')
    ax1.imshow(image1, interpolation='nearest', cmap = mpl.cm.gray)

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)
    plt.title('fft module')
    ax2.imshow(normalog(imgFinal), interpolation='nearest', cmap = mpl.cm.gray)

    Fini = mapping(image1, 30, 2)
    Fini = np.array(Fini)

##    import time
##    t = time.time()
##    for i in range(4):
##        mapping(image1, 30, 2)
##    print time.time() - t

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.axes.get_xaxis().set_visible(False)
    ax3.axes.get_yaxis().set_visible(False)
    plt.title('Oriantation Map')
    ax3.imshow(Fini[:,:,0], interpolation='nearest', cmap = mpl.cm.hsv)

    ax3 = fig.add_subplot(2, 3, 5)
    ax3.axes.get_xaxis().set_visible(False)
    ax3.axes.get_yaxis().set_visible(False)
    plt.title('Smooth Freq Map')
    ax3.imshow(smoothingFreq(Fini, block = 20, sigma = 2.), interpolation='nearest', cmap = mpl.cm.hsv)

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axes.get_xaxis().set_visible(False)
    ax6.axes.get_yaxis().set_visible(False)
    plt.title('Frequence Map')
    ax6.imshow(Fini[:,:,1], interpolation='nearest', cmap = mpl.cm.hsv)

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.axes.get_xaxis().set_visible(False)
    ax4.axes.get_yaxis().set_visible(False)
    plt.title('Proba Map')
    ax4.imshow(Fini[:,:,2], interpolation='nearest', cmap = mpl.cm.hsv)

    plt.show()
