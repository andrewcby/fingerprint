import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np

def varianceCrop(img, seuil = 100):
    xMin, xMax, yMin, yMax = 0, img.shape[0]-1, 0, img.shape[1]-1
    
    temp = np.average(img, axis=0)
    while temp[yMin]<seuil:
        yMin += 1

    while temp[yMax]<seuil:
        yMax -= 1

    temp = np.average(img, axis=1)
    while temp[xMin]<seuil:
        xMin += 1

    while temp[xMax]<seuil:
        xMax -= 1

    return (xMin, xMax, yMin, yMax)

def varianceMap(img, block):

    imgFinal = np.empty((len(img)//block, len(img[0])//block))
    
    for i in range(len(img)//block):
        for j in range(len(img[0])//block):

            zone = img[i*block:(i+1)*block,j*block:(j+1)*block]

            imgFinal[i][j] = ndimage.variance(zone)

    return imgFinal

def varianceMap_glisse(img, block):

    imgFinal = np.empty(img.shape)

    med = (block//2, block-(block//2))
    
    for i in range(len(img)):
        iMin, iMax = max(0, i-med[0]), min(len(img), i+med[1])
        for j in range(len(img[0])):
            jMin, jMax = max(0, j-med[0]), min(len(img[0]), j+med[1])

            zone = img[iMin:iMax,jMin:jMax]

            imgFinal[i][j] = ndimage.variance(zone)

    return imgFinal

# 3 time faster
def fastVarianceMap(img, block):

    imgFinal = np.empty((len(img)//block, len(img[0])//block))
    
    for i in range(len(img)//block):
        for j in range(len(img[0])//block):

            zone = img[i*block:(i+1)*block,j*block:(j+1)*block]

            imgFinal[i][j] = np.max(zone)-np.min(zone)

    return imgFinal

def fastVarianceMap_glisse(img, block):

    return ndimage.maximum_filter(img, block) - ndimage.minimum_filter(img, block)


# If you lauch this file alone :
if __name__ == '__main__':
    print "Laugtching as Script"

    image1 = ndimage.imread('D:\\Recherche\\DBB\\FVC2002\\DB2_B\\103_5.tif')

    sizeX, sizeY = image1.shape
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3 , sharex='col', sharey='row')

    ax1.imshow(image1,cmap=plt.cm.gray)

    ax2.imshow(ndimage.zoom(varianceMap(image1, 16), 16, order = 0),cmap=plt.cm.gray)
    ax4.imshow(ndimage.zoom(fastVarianceMap(image1, 16), 16, order = 0),cmap=plt.cm.gray)
    ax3.imshow(varianceMap_glisse(image1, 16),cmap=plt.cm.gray)
    ax5.imshow(fastVarianceMap_glisse(image1, 16),cmap=plt.cm.gray)

    plt.show()
