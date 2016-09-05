from scipy import ndimage
import numpy as np

def globalEnhacement(img):
    # The minimum value become 0
    img -= np.min(img)
    # And the maximum become 255
    img = 255. * img / np.max(img)

    return img

def localEnhacement(img, block):
    # Same as globalEnhacement but with
    # its neighbour
    imgFinal = np.empty_like(img)
    
    imgMin = ndimage.filters.minimum_filter(img, block)
    imgMax = ndimage.filters.maximum_filter(img, block)
    
    imgFinal = (img - imgMin) * 255.
    
    imgMax -= imgMin

    imgFinal /= imgMax

    # To deal with the part to uniform (include div by 0)
    imgFinal = np.where(imgMax<30, img, imgFinal)
    
    return imgFinal

# If you lauch this file alone :
if __name__ == '__main__':
    print "Laugtching as Script"
    import matplotlib.pyplot as plt

    image1 = ndimage.imread('D:\\Recherche\\DBB\\FVC2002\\DB3_B\\102_1.tif')
    image2 = ndimage.imread('D:\\Recherche\\DBB\\FVC2002\\DB2_B\\103_5.tif')
    image3 = ndimage.imread('3.tif')

    sizeX, sizeY = image1.shape
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex='col', sharey='row')

    ax2.imshow(image2,cmap=plt.cm.gray)
    ax3.imshow(image3,cmap=plt.cm.gray)
    ax1.imshow(image1,cmap=plt.cm.gray)

    ax5.imshow(localEnhacement(image2, 3),cmap=plt.cm.gray)
    ax6.imshow(localEnhacement(image3, 5),cmap=plt.cm.gray)
    ax4.imshow(localEnhacement(image1, 10),cmap=plt.cm.gray)

    plt.show()
