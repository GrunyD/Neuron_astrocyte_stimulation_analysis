import numpy as np
import matplotlib as mat
from PIL import Image


from consts import DEFAULT_GRID_SIZE

def n_active_rois(active_rois):
    return np.squeeze(np.apply_over_axes(np.sum, active_rois, [1,2]))

def shift_signal(signal:np.ndarray, shift:int):
    z = np.zeros((shift))
    new_signal = np.concatenate((signal, z))
    new_signal = np.roll(new_signal, shift)
    return new_signal[:-shift]

def tip_location_roi(tip_location:list, grid_size:int = DEFAULT_GRID_SIZE) -> list:
    # matplotlib.patches.Rectangel takes it in order x, y
    # subtract 0.5, otherwise the middle of the patch would be the grid crossing
    xy = [tip_location[1]//grid_size - 0.5, tip_location[0]//grid_size - 0.5]
    return xy

def normalize(arr):
    """
    Put the array into range 0-1
    """
    minimum = np.min(arr)
    arr = arr - minimum
    maximum = np.max(arr)
    arr = arr/maximum
    return arr

def IOU(heatmap1:np.ndarray, heatmap2:np.ndarray, threshold:float = None) -> float:
    """
    Intersection over union to quantify the spatial correlation
    """
    if threshold is not None:
        heatmap1 = heatmap1 > threshold
        heatmap2 = heatmap2 > threshold
    return np.sum(heatmap1*heatmap2)/(np.sum((heatmap1 + heatmap2) > 0))

def dice_score(heatmap1:np.ndarray, heatmap2:np.ndarray) -> float:
    """
    Dice score to quantify the spatial correlation
    """
    heatmap1  = normalize(heatmap1)
    heatmap2 = normalize(heatmap2)
    intersection = heatmap1*heatmap2
    dice = 2*np.sum(intersection)/(np.sum(heatmap1**2) + np.sum(heatmap2**2))
    # dice, _, _ = cv.EMD(heatmap1.astype(np.float32),heatmap2.astype(np.float32),cv.DIST_L1)
    return dice

def enhance_contrast(image):
    image -= np.min(image)
    image = np.array(255*image/np.max(image), dtype = np.uint8)
    hist,_ = np.histogram(image, bins = 256, range = (0, 255))
    cumsum = np.cumsum(hist[::-1])
    maximum = np.sum(cumsum < np.size(image)/70)
    image = image/(255 - maximum)
    np.clip(image,a_min = None, a_max = 1, out = image)
    
    return image 

def get_image(sequence, cell_type):
    rgb_image = np.zeros((sequence.shape[1], sequence.shape[2], 3))
    image = np.average(sequence, 0)
    image = enhance_contrast(image)
    index = 0 if cell_type == 'neurons' else 1
    rgb_image[:,:, index] = image
    return rgb_image

def overlap_image_map(image_sequence:np.ndarray, cell_type:str, heatmap:np.ndarray, vmin = 0, vmax = 10):
    cmap = mat.colormaps['plasma']
    im = cmap((heatmap-vmin)/vmax)*255
    im[:,:, 3] = 50 if cell_type == 'neurons' else 100
    im = Image.fromarray(im.astype(np.uint8))
    print(image_sequence.shape)
    im = im.resize((image_sequence.shape[1], image_sequence.shape[2]), 0)
    neurons_image = Image.fromarray((get_image(image_sequence, cell_type)*255).astype(np.uint8))
    neurons_image = neurons_image.convert("RGBA")
    return Image.alpha_composite(neurons_image, im)

def get_events(array:np.ndarray) -> list:
    """
    Array is only bool values
    This return indices where events starts and end

    Example:
    >>> array = [True, True, True, False, False, True, True, True, False]
    >>> split_bool(array)
    >>> [(0, 3), (5, 8)]
    """
    i = 0
    state = False
    parts = []
    start = None
    while i < len(array):
        if array[i]:
            if not state:
                start = i
                state = True
            else:
                pass
        else:
            if state:
                parts.append((start, i))
                state = False
            else:
                pass
        i += 1
    return parts





