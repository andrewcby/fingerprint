from scipy import ndimage
import numpy as np

import InterfacePreprocessing as InterPre
import MultiTools as Tool
import RegistrationSimple as RegSimple
import RegistrationBrute as RegBrute


def apercu(plot, image, nom = ''):
    ax1 = fig.add_subplot(2, 6, plot)
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    plt.title(nom)
    ax1.imshow(image, interpolation='nearest', cmap = mpl.cm.gray)

def registration(img1, img2, coord=False):
    # Detect the center of the two fingerprint
    center1, center2 = RegSimple.registration(img1, img2)

    # Choose the maximum ray possible around center1 without
    # exit the limit and extract this piece
    r = min(min(center1[0], len(img1)-center1[0]), min(center1[1], len(img1[0])-center1[1]))//4
    piece_img1 = img1[center1[0]-r:center1[0]+r, center1[1]-r:center1[1]+r]

    # Lauch bruteForce around the center to have a rough estimation
    # of the final registeration
    x, y, angle = RegBrute.bruteForce(piece_img1, img2, r, 3, 2, -50, 50, max(r, center2[0]-r), min(len(img2)-r, center2[0]+r), max(r, center2[1]-r), min(len(img2[0])-r, center2[1]+r))
    # And an other time, more precise, to have the real registeration
    x, y, angle = RegBrute.bruteForce(piece_img1, img2, r, 1, 1, angle-8, angle+8, max(0, center2[0]-5), min(len(img2), center2[0]+5), max(0, center2[1]-5), min(len(img2[0]), center2[1]+5))

    # Apply registeration
    rotate_img2 = ndimage.rotate(img2, angle, order = 1, reshape = False, cval = 255.)
    (center_image1, center_image2), pad = Tool.padCenters(img1, rotate_img2, center1, (x, y), dec=True)

    if coord:
        x += pad[0]
        y += pad[1]
        return ((center_image1, center_image2), (x, y))
    return (center_image1, center_image2)


# If you lauch this file alone :
if __name__ == '__main__':
    print "Laugtching as Script"
    
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    image1 = ndimage.imread('D:\\Recherche\\DBB\\CASIA-FingerprintV5(BMP)\\014\\L\\014_L3_1.bmp', mode = 'I')
    image2 = ndimage.imread('D:\\Recherche\\DBB\\CASIA-FingerprintV5(BMP)\\014\\L\\014_L3_2.bmp', mode = 'I')

    fig = plt.figure()

    apercu(1, image1, 'Image 1')
    apercu(7, image2, 'Image 2')

    ##

    seg_block = 10
    seg_image1 = InterPre.variance(image1, seg_block)
    seg_image2 = InterPre.variance(image2, seg_block)

    xMin, xMax, yMin, yMax = InterPre.varianceCrop(seg_image1)
    crop_image1 = image1[xMin:xMax,yMin:yMax]
    apercu(2, crop_image1, 'Decoupe')

    xMin, xMax, yMin, yMax = InterPre.varianceCrop(seg_image2)
    crop_image2 = image2[xMin:xMax,yMin:yMax]
    apercu(8, crop_image2, 'Decoupe')




    center_image1, center_image2 = registration(crop_image1, crop_image2)

    apercu(4, center_image1, 'Coherence') 
    apercu(10, center_image2, 'Coherence')




    apercu(5, center_image1 + center_image2, 'Coherence') 

    plt.show()
