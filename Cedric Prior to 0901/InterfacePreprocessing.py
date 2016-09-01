import Variance as Var
import EnhacementContrast as EnCon
import OrientationMap as OrMap
import OrFreqMapFFT as OrFrMapFFT
import RegistrationFFT as RegFFT
import RegistrationBrute as RegBru
import RegistrationSimple as RegSimp
import RegistrationCombinee as RegComb
import MultiTools as Tool

regSize = None
        
def contrast(img, block=8, total=False):
    """
        All to enhance the constrast of an image
 
        Two algorythm are implemented : one who change
        the constrast of the entire image (like photoshop
        or other), and the other (really appropriate for
        fingerprint, so always active) ajust the constrast of each
        pixel according to its neighbors (in a radius of r pixels)

        If total are true, the image will automaticly have all here
        value between 0 and 255
 
        :param img: A numpy array who represent the image
        :param block: The size (in px) of the neighbors (a block of block*block)
        :param total: To say if you want to ajust the entiere constrast
        in place of the local
        :return: The same numpy image but with the better contrast than
        before
    """
    
    if total:
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
        with the same shape than the entry
        :return: A (couple if coherence True) of numpy array
    """
    
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

def frequency(img, block = 15, step = 2, smoothFreq=False, smoothFreqBlock=20, smoothFreqSigma=1.5):
    """
        Generate an frequence map (and can smooth it)
 
        :param img: A numpy array who represent the image
        :param block: The size (in px) of the block (of size block*block)
        :param step: step of the moving windows
        :param smoothFreq: if you want to smooth the frequency map
        :param smoothFreqBlock: Size of the block for the smoothing
        :param smoothFreqSigma: Sigma because we use a Gauss kernel
        :return: A numpy array of frequency
    """

    maps = OrFrMapFFT.mapping(img, block, step, False, True, smoothFreq)

    if smoothFreq:
        return OrFrMapFFT.smoothingFreq(maps, smoothFreqBlock, smoothFreqSigma)
    return [[j[1] for j in i] for i in maps]

def variance(img, block=10, moving=False, fast=False, resize=True):
    """
        Generate a variation map who permit to extract the fingerprint
        of a useless part (totaly whrite for exemple)
 
        :param img: A numpy array who represent the image
        :param block: The size (in px) of the neighbors (a block of block*block)
        :param moving: If you want to use a moving windows
        :param fast: Use an ather algorithm 3 time faster than variation
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
        
def varianceCrop(img, seuil = 100):
    return Var.varianceCrop(img, seuil)

def registrationFFT(img1, img2):
    """
        Ajust images between us (rotation and translation)
 
        :param img1 and img2: Two numpy array with the same shape
        :return: The two images ajusted (and crop, so the size are
        the same for the two images, but could be smaller than the entry)
        
        .. todo:: Permit to choose to return a same size image or not
                  Permit to have a transparent background
    """

    if img1.shape != img2.shape:
        raise Exception('The two images must have the same size')
    
    if img1.shape != regSize:
        RegFFT.pre_logpolar(img1.shape)

    return RegFFT.registration(img1, img2)

def registrationBrute(img1, img2, r=None, cropLess=False):
    """
        Ajust images between us (rotation and translation)
 
        :param img1 and img2: Two numpy array
        :return: The two images ajusted (and crop, so the size are
        the same for the two images, but could be different than the entry)
        
    """

    return RegBru.registration(img1, img2, r, cropLess)

def registrationSimple(img1, img2, cropLess=False):
    """
        Ajust images between us (translation)
        Usefull when images are differents
 
        :param img1 and img2: Two numpy array
        :return: The two images ajusted (and crop, so the size are
        the same for the two images, but could be different than the entry)
        
    """

    return RegSim.registration(img1, img2, cropLess)
