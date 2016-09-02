from scipy import ndimage
import numpy as np

def globalEnhacement(img):

    img = img - np.min(img)
    img = 255. * img / np.max(img)

    return img

def localEnhacement(img, block):

    imgFinal = np.empty_like(img)

    imgMin = ndimage.filters.minimum_filter(img, block)
    imgMax = ndimage.filters.maximum_filter(img, block)
    
    imgMaxMin = imgMax - imgMin
    del imgMax
    
    imgFinal = (img - imgMin) * 255.
    del imgMin
    
    for i in range(len(img)):
        for j in range(len(img[0])):
            
            maxmin = imgMaxMin[i][j]

            if(maxmin<20):
                imgFinal[i][j] = img[i][j]
            else:
                imgFinal[i][j] /= maxmin

    return imgFinal


# If you lauch this file alone :
if __name__ == '__main__':
    print "Laugtching as Script"

    import matplotlib.pyplot as plt

    image1 = ndimage.imread('D:\\Recherche\\DBB\\FVC2002\\DB3_B\\102_1.tif')
    image2 = ndimage.imread('D:\\Recherche\\DBB\\PERSO\\14_3.png')
    image3 = ndimage.imread('D:\\Recherche\\DBB\\PERSO\\2_2.png')

    sizeX, sizeY = image1.shape
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex='col', sharey='row')

    ax2.imshow(image2,cmap=plt.cm.gray)
    ax3.imshow(image3,cmap=plt.cm.gray)
    ax1.imshow(image1,cmap=plt.cm.gray)

    ax5.imshow(localEnhacement(image2, 3),cmap=plt.cm.gray)
    ax6.imshow(localEnhacement(image3, 5),cmap=plt.cm.gray)
    ax4.imshow(localEnhacement(image1, 10),cmap=plt.cm.gray)

    plt.show()
