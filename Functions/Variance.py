from scipy import ndimage
import numpy as np

def varianceCrop(img, seuil=10):
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
    
    return ndimage.filters.generic_filter(img, np.std, size=block)

# 50 time faster
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
    import matplotlib.pyplot as plt

    image = ndimage.imread('D:\\Recherche\\DBB\\FVC2002\\DB2_B\\103_8.tif')

    sizeX, sizeY = image.shape
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3 , sharex='col', sharey='row')

    ax1.imshow(image,cmap=plt.cm.gray)
    ax4.imshow(image,cmap=plt.cm.gray)

    ax2.imshow(ndimage.zoom(varianceMap(image, 16), 16, order = 0),cmap=plt.cm.gray)
    ax3.imshow(varianceMap_glisse(image, 16),cmap=plt.cm.gray)
    
    ax5.imshow(ndimage.zoom(fastVarianceMap(image, 16), 16, order = 0),cmap=plt.cm.gray)
    ax6.imshow(fastVarianceMap_glisse(image, 16),cmap=plt.cm.gray)

    plt.show()
