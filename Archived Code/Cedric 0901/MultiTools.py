import numpy as np
from scipy import ndimage
from PIL import Image

def showImg(img):
    img -= np.min(img)
    img /= (np.max(img) / 255.)
    Image.fromarray(img.astype(np.int32), 'I').show()
    
def smoothingAngle(img, sigma):
    img = img * 2.0
    imgSin = np.sin(img)
    imgCos = np.cos(img)

    imgSin = ndimage.filters.gaussian_filter(imgSin, sigma)
    imgCos = ndimage.filters.gaussian_filter(imgCos, sigma)

    return 0.5 * np.arctan2(imgSin, imgCos)

def zoom(img, shape):
    #ndimage.zoom(orientMap, 20, order=0)
    
    x, y = img.shape

    r = shape[0]//x
    
    bigImg = np.zeros(shape, dtype=img.dtype)

    for i in range(x):
        for j in range(y):
            bigImg[i*r:(i+1)*r,j*r:(j+1)*r] = img[i][j]

    for i in range(x*r, shape[0]):
        bigImg[i] = bigImg[x*r-1]

    for j in range(y*r, shape[1]):
        bigImg[:,j] = bigImg[:,y*r-1]

    return bigImg

def padMax(img1, img2, constant=255):
    
    difX = img1.shape[0] - img2.shape[0]
    if difX<0:
        difX = abs(difX)
        img1 = np.pad(img1, ((difX//2, difX-difX//2), (0,0)), mode='constant', constant_values=constant)
    elif difX>0:
        img2 = np.pad(img2, ((difX//2, difX-difX//2), (0,0)), mode='constant', constant_values=constant)
        
    difY = img1.shape[1] - img2.shape[1]
    if difY<0:
        difY = abs(difY)
        img1 = np.pad(img1, ((0,0), (difY//2, difY-difY//2)), mode='constant', constant_values=constant)
    elif difY>0:
        img2 = np.pad(img2, ((0,0), (difY//2, difY-difY//2)), mode='constant', constant_values=constant)

    return img1, img2

def padCenters(img1, img2, center1, center2, cropLess=False, lesDeux=False):
    xCen, yCen = center1
    x, y = center2

    xDif, yDif = xCen - x, yCen - y

    center_img2 = np.pad(img2, ((max(0, xDif), 0), (max(0, yDif), 0)), mode='constant', constant_values=255)
    center_img1 = np.pad(img1, ((max(0, -xDif), 0), (max(0, -yDif), 0)), mode='constant', constant_values=255)

    xMin, yMin = abs(xDif), abs(yDif)

    xDif, yDif = center_img2.shape[0] - center_img1.shape[0], center_img2.shape[1] - center_img1.shape[1]

    center_img2 = np.pad(center_img2, ((0, max(0, -xDif)), (0, max(0, -yDif))), mode='constant', constant_values=255)
    center_img1 = np.pad(center_img1, ((0, max(0, xDif)), (0, max(0, yDif))), mode='constant', constant_values=255)

    xMax, yMax = abs(xDif), abs(yDif)

    if cropLess:
        return (center_img1[xMin:-xMax,yMin:-yMax], center_img2[xMin:-xMax,yMin:-yMax])
    if lesDeux:
        return ((center_img1, center_img2), (xMin,xMax,yMin,yMax))
    return (center_img1, center_img2)

def binarise(img, med=None):
    if med==None:
        med = np.average(img)
        
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] < med:
                img[i][j] = 0
            else:
                img[i][j] = 1
