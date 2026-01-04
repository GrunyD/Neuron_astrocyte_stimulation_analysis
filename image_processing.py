import numpy as np
import matplotlib as mat
from PIL import Image
import cv2 as cv
import qim3d

from consts import DEFAULT_GRID_SIZE
from input_output import rois_indices

def get_image(image:np.ndarray, cell_type:str, enhanced:bool = False):

    rgb_image = np.zeros((image.shape[0], image.shape[1], 3))
    if not enhanced:
        rgb_image = np.zeros((image.shape[1], image.shape[2], 3))
        image = np.average(image, 0)
        image = enhance_contrast(image)
    index = 0 if cell_type == 'neurons' else 1
    rgb_image[:,:, index] = image
    return rgb_image

def enhance_contrast(image):
    image -= np.min(image)
    image = np.array(255*image/np.max(image), dtype = np.uint8)
    hist,_ = np.histogram(image, bins = 256, range = (0, 255))
    cumsum = np.cumsum(hist[::-1])
    maximum = np.sum(cumsum < np.size(image)/70)
    image = image/(255 - maximum)
    np.clip(image,a_min = None, a_max = 1, out = image)
    
    return image

def prepare_image(image:np.ndarray):
    if image.ndim == 3:
        image  = enhance_contrast(np.average(image, axis = 0))
    image = image*255
    image =image.astype(np.uint8)
    return cv.medianBlur(image,5)


def segment_neurons(image:np.ndarray):
    image = prepare_image(image)
    image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                         cv.THRESH_BINARY,15, -15)
    kernel = np.array((
        (0, 1, 0),
        (1, 1, 1),
        (0, 1, 0)
        ),dtype = np.uint8)
    opening = cv.morphologyEx(image, cv.MORPH_OPEN, kernel, borderValue=0)
    # Gets rid of big structures that I couldnt filter out any other way
    res, num = qim3d.segmentation.watershed(opening, 0)
    for i in range(1, num):
        pos = np.where(res == i)
        lu = np.array((pos[0].min(), pos[1].min()))
        rb = np.array((pos[0].max(), pos[1].max()))
        area = np.prod(rb - lu)
        if area > 200:
            opening[res == i] = 0

    return opening


def segment_astrocytes(image:np.ndarray):
    image = prepare_image(image)
    image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C, 
                                 cv.THRESH_BINARY,31, -13)
    kernel = np.array((
        (0, 1, 1, 1, 0),
        (1, 1, 1, 1, 1),
        (1, 1, 1, 1, 1),
        (1, 1, 1, 1, 1),
        (0, 1, 1, 1, 0)
        ),dtype = np.uint8)
    opening = cv.morphologyEx(image, cv.MORPH_OPEN, kernel, borderValue=0)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)

    return closing

def rois_from_segmentation(segmentation:np.ndarray, 
                           grid_size:int = DEFAULT_GRID_SIZE):
    """
    Returns percentage of how many pixels were recognized as 
    'soma' in given ROI. 
    """

    seg_shape = np.array(segmentation.shape)
    shape = seg_shape//grid_size + ((seg_shape%grid_size)>0)
    res = np.zeros(shape)
    for i, j in rois_indices(segmentation.shape, grid_size):
        upper_limit_row = (i+1)*grid_size
        if upper_limit_row >= segmentation.shape[0]:
            upper_limit_row = None
        upper_limit_col = (j+1)*grid_size
        if upper_limit_col >= segmentation.shape[1]:
            upper_limit_col = None
        res[i, j] = np.sum((segmentation>0)[i*grid_size:upper_limit_row, j*grid_size:upper_limit_col])
    
    res = 100 *res / grid_size**2
    return res
