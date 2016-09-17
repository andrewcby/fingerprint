# -*- coding: cp1252 -*-
import Variance as Var
import EnhacementContrast as EnCon
import EnhacementGabor as EnGab
import OrientationMap as OrMap
import OrFreqMapFFT as OrFrMapFFT
import RegistrationFFT as RegFFT
import RegistrationBrute as RegBru
import RegistrationSimple as RegSimp
import RegistrationCombinee as RegComb
import MultiTools as Tool

import numpy as np

regSize = None
gabPrec = None
        
def contrast(img, block=8, Global=False):
    """
        All to enhance the constrast of an image
 
        Two algorythm are implemented : one who change
        the constrast of the entire image (like photoshop
        or other), and the other (really appropriate for
        fingerprint) ajust the constrast of each pixel
        according to its neighbors (in a radius of r pixels)
 
        :param img: A numpy array who represent the image
        :param block: The size (in px) of the neighbors (a block of block*block)
        :param Global: To say if you want to ajust the entiere constrast
        in place of the local
        :return: The same numpy image but with the better contrast than
        before
        
    """
    
    if Global:
        return EnCon.globalEnhacement(img)

    return EnCon.localEnhacement(img, block)

def orientation(img, block=10, moving=True, coherence=False, resize=True):
    """
        Generate an orientation map (and a coherence map in option)
 
        :param img: A numpy array who represent the image
        :param block: The size (in px) of the block (size of block*block)
        :param moving: If you want to use a moving windows
        :param coherence: For returning the coherence map at
        the same time that the orientation map
        :param resize: To return an orientation map (and coherence)
        with the same shape than the entry (useless with moving=True)
        :return: A (couple if coherence True of) numpy array
        
    """
    
    img = img.astype(np.int32)

    if moving:
        return OrMap.generate_glissant(img, block, coherence)
    
    if not resize:
        return OrMap.generate(img, block, coherence)

    if coherence:
        orTemp, coTemp = OrMap.generate(img, block, coherence)
        return (Tool.zoom(orTemp, img.shape), Tool.zoom(coTemp, img.shape))
    else:
        orTemp = OrMap.generate(img, block, coherence)
        return Tool.zoom(orTemp, img.shape)

def frequency(img, block=15, step=2, smoothFreq=False, smoothFreqBlock=20, smoothFreqSigma=1.5):
    """
        Generate an frequence map (and can smooth it)
 
        :param img: A numpy array who represent the image
        :param block: The size (in px) of the block (of size block*block)
        :param step: step for the moving windows
        :param smoothFreq: if you want to smooth the frequency map
        :param smoothFreqBlock: Size of the block for the smoothing
        :param smoothFreqSigma: Sigma of the Gauss filter for the smoothing
        :return: A numpy array of frequency
        
    """

    maps = OrFrMapFFT.mapping(img, block, step, False, True, smoothFreq)
    
    if smoothFreq:
        tmp = OrFrMapFFT.smoothingFreq(maps, smoothFreqBlock, smoothFreqSigma)
        if step>1:
            return Tool.zoom(tmp, img.shape)
        return tmp
    
    if step>1:
        return Tool.zoom(maps, img.shape)
    
    return np.array(maps)[:,:,1]

def variance(img, block=10, moving=True, fast=False, resize=True):
    """
        Generate a variation map who permit to extract the fingerprint
        from the background
 
        :param img: A numpy array who represent the image
        :param block: The size (in px) of the neighbors (a block of block*block)
        :param moving: If you want to use a moving windows
        :param fast: Use an other algorithm 50 time faster than variation
        but results are a little differents
        :param resize: To return an variation map with the same shape than
        the entry
        :return: A numpy array
        
    """
    
    if moving:
        if fast:
            return Var.fastVarianceMap_glisse(img, block)
        else:
            return Var.varianceMap_glisse(img, block)
    else:
        if fast:
            imgVar = Var.fastVarianceMap(img, block)
        else:
            imgVar =  Var.varianceMap(img, block)

    if resize:
        imgVar = Tool.zoom(imgVar, img.shape)

    return imgVar

def gabor(img, precision=15, freq=(0.078, 0.157, 0.01), sigma=(6, 9, 1), extract=False):
    """
        Apply Gabor filters to obtained an enhacement of the
        image (less scratch etc...).
 
        :param img: A numpy array who represent the image
        :param precision: Number of step to explore the 360° of possible
        angles (biger is slower)
        :param freq: A n-upplet composed of the minimum and maximum
        frequency possible and the step to pass from the min to max
        :param sigma: Same things that freq but with Sigma
        :param extract: If you want to remove the background (useless
        if you don't binarise the image after)
        :return: The enhacent image (where value are 0 or 1)
        
    """

    if gabPrec != precision:
        EnGab.generateKernel(precision, freq, sigma)

    return EnGab.enhacementGabor(img, extract)

def varianceCrop(img, seuil=10):
    """
        Return two coordinate witch create a rectangle
        around the fingerprint.

        :param img: The variance map of the image to
        extract
        :return: (xMin, xMax, yMin, yMax)
        
    """
    
    return Var.varianceCrop(img, seuil)

def registrationFFT(img1, img2):
    """
        Ajust images between us (rotation and translation) by using
        the phase correlation
        Really fast but have dificulties depending of the image
 
        :param img1 and img2: Two numpy array with the same shape
        :return: The two images ajusted (and crop, so the size are
        the same for the two images, but could be smaller than the entry)
        
    """

    if img1.shape != img2.shape:
        raise Exception('The two images must have the same size')
    
    if img1.shape != regSize:
        RegFFT.pre_logpolar(img1.shape)

    return RegFFT.registration(img1, img2)

def registrationBrute(img1, img2, r=None, cropLess=False):
    """
        Ajust images between us (rotation and translation)
        by testing all the possibilities
        Really slow but good results
 
        :param img1 and img2: Two numpy array
        :return: The two images ajusted (and crop, so the size are
        the same for the two images, but could be different than the entry)
        
    """

    return RegBru.registration(img1, img2, r, cropLess)

def registrationSimple(img1, img2, cropLess=False):
    """
        Ajust images between us (translation) by detecting
        its center.
        Basic but really fast and good
        Usefull when images are differents
 
        :param img1 and img2: Two numpy array
        :return: The two images ajusted (and crop, so the size are
        the same for the two images, but could be different than the entry)
        
    """

    return RegSim.registration(img1, img2, cropLess)


def RegistrationCombinee(img1, img2, coord=False):
    """
        Ajust images between us (translation and translation)
        Using registrationSimple for a first and fast displacement
        and after that use registrationBrute arround this position
        Work with all pairs (match and mismatch) fast with good results.
 
        :param img1 and img2: Two numpy array
        :return: The two images ajusted (and crop, so the size are
        the same for the two images, but could be different than the entry)
        
    """

    return RegComb.registration(img1, img2, coord)
