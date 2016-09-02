from scipy import ndimage
import numpy as np

import InterfacePreprocessing as InterPre
import MultiTools as Tool


def simple(img):
    new_img = np.zeros(img.shape)
    for i in range(1, len(img)-1):
        for j in range(1, len(img[0])-1):
            nb = img[i+1][j]+img[i-1][j]+img[i][j-1]+img[i][j+1]
            if nb>2:
                new_img[i][j] = 1
    return new_img


def registration(img1, img2):
    seg_block = 10
    or_img1 = abs(InterPre.orientation(img1, 30, coherence=False))
    or_img2 = abs(InterPre.orientation(img2, 30, coherence=False))

    coh_img1 = InterPre.variance(or_img1, 5, moving=True, fast=True)
    coh_img2 = InterPre.variance(or_img2, 5, moving=True, fast=True)

    ## VarMap binaire
    seg_image1 = InterPre.variance(img1, seg_block, resize=False)
    seg_image2 = InterPre.variance(img2, seg_block, resize=False)

    Tool.binarise(seg_image1, 700)
    Tool.binarise(seg_image2, 700)
    
    seg_image1 = simple(seg_image1)
    seg_image1 = simple(seg_image1)
    seg_image1 = simple(seg_image1)
    
    seg_image2 = simple(seg_image2)
    seg_image2 = simple(seg_image2)
    seg_image2 = simple(seg_image2)
    
    seg_image1 = Tool.zoom(seg_image1, coh_img1.shape)
    seg_image2 = Tool.zoom(seg_image2, coh_img2.shape)

    ## Registration
    def maximum(img):
        for i in range(len(img)):
            for j in range(len(img[0])):
                if img[i][j] > 1:
                    return (i, j)
        return np.unravel_index(np.argmax(img), img.shape)
    
    center1 = maximum(seg_image1*coh_img1)
    center2 = maximum(seg_image2*coh_img2)

    return (center1, center2)

# If you lauch this file alone :
if __name__ == '__main__':
    print "Laugtching as Script"
    
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    
    fig = plt.figure()

    def apercu(plot, image, nom = ''):
        ax1 = fig.add_subplot(2, 6, plot)
        ax1.axes.get_xaxis().set_visible(False)
        ax1.axes.get_yaxis().set_visible(False)
        plt.title(nom)
        ax1.imshow(image, interpolation='nearest', cmap = mpl.cm.gray)


    image1 = ndimage.imread('D:\\Recherche\\DBB\\FVC2002\\DB2_B\\102_1.tif', mode = 'I')
    image2 = ndimage.imread('D:\\Recherche\\DBB\\FVC2002\\DB2_B\\109_1.tif', mode = 'I')

    ## Raw Images
    apercu(1, image1, 'Image 1')
    apercu(7, image2, 'Image 2')

    ## Variance maps and cropping
    seg_block = 10
    seg_image1 = InterPre.variance(image1, seg_block)
    seg_image2 = InterPre.variance(image2, seg_block)

    xMin, xMax, yMin, yMax = InterPre.varianceCrop(seg_image1)
    crop_image1 = image1[xMin:xMax,yMin:yMax]
    apercu(2, crop_image1, 'Decoupe')

    xMin, xMax, yMin, yMax = InterPre.varianceCrop(seg_image2)
    crop_image2 = image2[xMin:xMax,yMin:yMax]
    apercu(8, crop_image2, 'Decoupe')
    
    apercu(3, InterPre.variance(abs(InterPre.orientation(crop_image1, 30, coherence=False)), 5, moving=True, fast=True), 'Commun')
    apercu(9, InterPre.variance(abs(InterPre.orientation(crop_image2, 30, coherence=False)), 5, moving=True, fast=True), 'Commun')

    ## Registration simple multiple
    center1, center2 = registration(crop_image1, crop_image2)
    center_image1, center_image2 = Tool.padCenters(crop_image1, crop_image2, center1, center2)
    
    apercu(4, crop_image1[center1[0]-30:center1[0]+30, center1[1]-30:center1[1]+30], 'Commun')
    apercu(10, crop_image2[center2[0]-30:center2[0]+30, center2[1]-30:center2[1]+30], 'Commun')

    apercu(6, center_image1*0.5 + center_image2*0.5, 'Coherence')
    apercu(12, center_image2, 'Coherence')

    plt.show()
