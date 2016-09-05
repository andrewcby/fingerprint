from scipy import ndimage
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

import MultiTools as Tool

def bruteForce(center_img1, img2, r, pasOr, pasTr, orMin, orMax, xMin, xMax, yMin, yMax):
    corMin = -1
    x, y, angle = 0, 0, 0
    # Test each possible orientations and translation
    # to find the best corelation
    for k in  range(orMin, orMax, pasOr):
        rotate_image2 = ndimage.rotate(img2, k, order = 0, reshape = False, cval = 255.)
        for i in range(xMin, xMax, pasTr):
            for j in range(yMin, yMax, pasTr):
                center_img2 = rotate_image2[i-r:i+r,j-r:j+r]

                # Calculate the correlation
                temp1 = center_img1 - np.average(center_img1)
                temp2 = center_img2 - np.average(center_img2)
                temp = np.multiply(temp1, temp2)
                temp = np.sum(temp)/(temp2.shape[0] * temp2.shape[1])
                correl = temp

                if corMin<correl:
                    corMin = correl
                    x, y, angle = i, j, k
                
##                correl = np.sum(abs(center_img1 - center_img2))
##                if corMin>correl or corMin<0:
##                    corMin = correl
##                    x, y, angle = i, j, k
    return x, y, angle


def registration(img1, img2, r=None, cropLess=False, lesDeux=False):
    if r==None:
        r = min(min(img1.shape), min(img2.shape))//4

    # Extract a part of the original image
    xCen, yCen = img1.shape[0]//2, img1.shape[1]//2
    center_img1 = img1[xCen-r:xCen+r,yCen-r:yCen+r]

    # Registeration with a smaller step each time
    xMin, xMax, yMin, yMax = r, len(img2)-r, r, len(img2[0])-r
    x, y, angle = bruteForce(center_img1, img2, r, 6, 4, -40, 40, xMin, xMax, yMin, yMax)
    x, y, angle = bruteForce(center_img1, img2, r, 2, 2, angle-6, angle+6, max(xMin, x-4), min(xMax, x+4), max(yMin, y-4), max(yMax, y+4))
    x, y, angle = bruteForce(center_img1, img2, r, 1, 1, angle-2, angle+2, max(xMin, x-2), min(xMax, x+2), max(yMin, y-2), max(yMax, y+2))

    ## Registration final
    xCen, yCen = img1.shape[0]//2, img1.shape[1]//2

    return Tool.padCenters(img1, img2, (xCen, yCen), (x, y), cropLess, lesDeux)


# If you lauch this file alone :
if __name__ == '__main__':
    print "Laugtching as Script"

    import InterfacePreprocessing as InterPre
    
    fig = plt.figure()

    def apercu(plot, image, nom = ''):
        ax1 = fig.add_subplot(2, 6, plot)
        ax1.axes.get_xaxis().set_visible(False)
        ax1.axes.get_yaxis().set_visible(False)
        plt.title(nom)
        ax1.imshow(image, interpolation='nearest', cmap = mpl.cm.gray)


    image1 = ndimage.imread('D:\\Recherche\\DBB\\FVC2002\\DB2_B\\105_1.tif', mode = 'I')
    image2 = ndimage.imread('D:\\Recherche\\DBB\\FVC2002\\DB2_B\\104_3.tif', mode = 'I')

    ## Raw Images
    apercu(1, image1, 'Image 1')
    apercu(7, image2, 'Image 2')

    ## Variance maps
    seg_block = 10
    seg_image1 = InterPre.variance(image1, seg_block)
    seg_image2 = InterPre.variance(image2, seg_block)

    apercu(2, seg_image1, 'Variance')
    apercu(8, seg_image2, 'Variance')

    ## Cropping of each images (based on variance map)
    xMin, xMax, yMin, yMax = InterPre.varianceCrop(seg_image1)
    crop_image1 = image1[xMin:xMax,yMin:yMax]
    apercu(3, crop_image1, 'Decoupe')

    xMin, xMax, yMin, yMax = InterPre.varianceCrop(seg_image2)
    crop_image2 = image2[xMin:xMax,yMin:yMax]
    apercu(9, crop_image2, 'Decoupe')

    ## Registration simple multiple
    center_image1, center_image2 = registration(crop_image1, crop_image2, 30)

    apercu(4, center_image1, 'Commun')
    apercu(10, center_image2, 'Commun')

    ## Orientation Map
    OMap_image1 = InterPre.orientation(center_image1, 30, coherence=False)
    OMap_image2 = InterPre.orientation(center_image2, 30, coherence=False)

    apercu(12, (abs(OMap_image1) - abs(OMap_image2)), 'Orientation Diff')

    apercu(5, abs(OMap_image1), 'Orient Map')
    apercu(11, abs(OMap_image2), 'Orient Map')

    ## Diff
    apercu(6, (center_image1/2 + center_image2/2), 'Images Superposed')

    plt.show()
