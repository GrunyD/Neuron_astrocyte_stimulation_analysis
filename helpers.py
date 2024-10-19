#Loading files
import os
from math import factorial
import io

import tifffile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
from scipy import sparse

# For video
import PIL
import cv2 as cv
import tqdm

RESULTS_FOLDER = ''
DATA_FOLDER = ''

DEFAULT_GRID_SIZE = 8

########################
# Functions for getting information from the filename
range_size = lambda shape, grid_size: shape//grid_size + int(shape%grid_size > 0)

file_by_number = lambda number, folder: [file for file in os.listdir(folder) if file.startswith(F'{str(number)}_')  and file.endswith('.tiff')][0]

is_ttx = lambda filename: 'ttx' in list(filename.split(' '))[-1]

exp_title = lambda number, current, ttx: F'Experiment number {number}: {current}uA with{" " if ttx else "out "}ttx'

def current(filename):
    first_part = list(filename.split(' '))[0]
    second_part = list(first_part.split('_'))[-1]
    current = int(second_part[:-2])
    return current

def experiment_number(filename):
    first_part = list(filename.split(' '))[0]
    return int(list(first_part.split('_'))[0])

########################
# Loading data
def get_data(filename:str, full_path:bool = False) -> tuple:
    """
    Loads data and returns two time sequences. Expects that the channel axis is axis number 2

    Parameters:
    -----------
    - filename: name of the file in the data_folder
    - full_path: If false looks for the file in DATA_FOLDER. If True expects that
        filename is full path to the file

    Retruns:
    -----------
        (Channel 1, Channel 2, both numpy arrays)
    """
    filename_path = filename if full_path else os.path.join(DATA_FOLDER, filename)
    image = tifffile.imread(filename_path)
    print("Data shape: ", image.shape)
    return image[:, 0], image[:, 1]

########################
# Iterator through indices based on grid size
def rois_indices(image_shape:(tuple), grid_size:int = DEFAULT_GRID_SIZE):
    for i in range(range_size(image_shape[0], grid_size)):
        for j in range(range_size(image_shape[1], grid_size)):
            yield i, j

def normalize(a:np.ndarray, mode:str = 'median', range_for_baseline:int = None):
    """
    Returns (a-F0)/F0
    computes F0 across axis = 0

    Parameters:
    ------------
    a: (np.ndarray) array to be normalized accros frames
    mode: (str) how to compute F0. Either 'mean' or 'median'
    range_for_baseline: (int) in case the baseline is not computed over the whole signal
        but only the first part, this defines (in frames) how long is the first part
    """
    match mode:
        case 'median':
            func = np.median
        case 'mean':
            func = np.mean
        case _:
            raise NotImplementedError(F"Mode has to be 'mean' or 'meadian'. You used '{mode}'")
    
    if range_for_baseline is not None:
        F0 = func(a[:range_for_baseline], axis = 0) 
    else:
        F0 = func(a, axis = 0)
    
    return (a-F0)/F0

########################
# Drawing plots 
def draw_full_grid(image:np.ndarray, grid_size:int = DEFAULT_GRID_SIZE, linewidth:float = 0.2, color:str = 'white', cmap = 'grey'):
    """
    Parameters:
    ---------------
    image: (np.ndarray) image to be drawn on
    grid_size: (int) how many pixels are in a rectangle
    linewidth: (float) linewidth of the grid (using matplotlib)
    color: (str) grid color (using matplotlib)
    """

    fig, ax = plt.subplots()
    ax.imshow(image, cmap = cmap)
    for i, j in rois_indices(image.shape, grid_size):
        rect = ptch.Rectangle((j*grid_size, i*grid_size), grid_size, grid_size, edgecolor = color, facecolor = 'none', linewidth = linewidth)
        ax.add_patch(rect)
    plt.show()

def draw_active_rois(image:np.ndarray, active_rois:np.ndarray, grid_size:int = DEFAULT_GRID_SIZE, linewidth:float = 0.6, color:str = 'red', cmap ='grey', ax = None, **kwargs):
    """
    Draws only those ROIs which were deemed as active/responsive

    Args:
    ------
    - image: image to draw on
    - active rois: 2D array of True or False to know which are active
    - grid_size: How many pixels is one grid
    - linewidth: How wide is the line which draws the active region
    - color: color of the rectangle, goes into pyplot functions so use pyplot-like arguments
    - cmap: which cmap to use to display the image (which is thought to be grayscale)
    - ax: if this image display is just a subplot of bigger figure, then it should recieve the subplot's ax
        If only displayed in notebook, it can be None
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(image, cmap = cmap)
    for i, row in enumerate(active_rois):
        for j, is_active in enumerate(row):
            if bool(is_active):
                rect = ptch.Rectangle((j*grid_size, i*grid_size), grid_size, grid_size, edgecolor = color, facecolor = 'none', linewidth = linewidth)
                ax.add_patch(rect)
    if ax is None:
        plt.show()

def plot_roi_spatial_activity(roi_activity:np.ndarray, exp_info:tuple, cell_type:str):
    fig, ax = plt.subplots()
    image = np.sum(roi_activity, axis = 0)/roi_activity.shape[0]
    axes_image = ax.imshow(image)
    ax.set_title(F'Relative RoI activity of {cell_type}\n{exp_title(*exp_info)}')
    fig.colorbar(axes_image)
    ax.set_axis_off()
    fig.savefig(F'{RESULTS_FOLDER}{exp_info[0]}_roi_spatial_activity_{cell_type}.png')

def plot_roi_temporal_activity(roi_activity_n:np.ndarray, roi_activity_a:np.ndarray, exp_info:tuple, grid_size:int = DEFAULT_GRID_SIZE, sample_freq:float = 2, save_fig:str = False):
    """
    roi_activity_n: How many ROIs were active at given time point for neurons
    roi_activity_a: How many ROIs were active at gievn time point for astrocytes
    exp_info: tuple with information about experiment
        expected as (experiment_number: int, electrical current: int, if_ttx_was_used: bool)
        if None, it won't be displayed
    grid_size: 
    sample_freqeuncy: Sampling frequency in Hz to display time axis in correct units
    save_fig_name: If True it will be saved in RESULTS folder
        Defaults to False, which will display the image
    """
    t = np.arange(len(roi_activity_n))//sample_freq
    fig, ax = plt.subplots()
    ax.plot(t, roi_activity_n, 'r')
    ax.plot(t, roi_activity_a, 'g')
    ax.legend(['Neurons', 'Astrocytes'])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(F'Number of active RoIs\ngrid size: {grid_size}')
    ax.set_title(F'Active RoIs in time\n{exp_title(*exp_info) if exp_info is not None else ""}')
    if save_fig:
        fig.savefig(F'{RESULTS_FOLDER}{exp_info[0]}_roi_temporal_activity.png')
    else:
        plt.show()

########################
# Thresholds
def MAD(a:np.ndarray, median:float = None):
    """
    Computes Median Absolute Deviation

    a: array
    median: in case it was already computed so we don't have to compute it again
    """
    if median is None:
        median = np.median(a, axis = 0)
    return np.median(np.abs(a - median), axis = 0)

def get_thresholds(a:np.ndarray, scalar:int = 1, frames_before_first_impulse:int = 16) -> float:
    """
    a: array, signal
    scalar: (int) How to scale the MAD   
    frames_before_first_impulse: The median shouldn't include signal with responses to the impulses
        because they then become included in normal behaviour 
        if None, whole signal will be taken
        Default is 16, as most of our experiments has that value
    """
    if frames_before_first_impulse is not None:
        a = a[:frames_before_first_impulse, :, :]
    median = np.median(a, axis = 0)
    return median + scalar*MAD(a, median)

########################
# Signal extraction
def image_to_roi_signals(image:np.ndarray, grid_size:int = DEFAULT_GRID_SIZE):
    """
    Turns the image to signals, based on how big is one ROI

    Parameters:
    ---------------
    image: (time, dim1, dim2) np.ndarray
    grid_size: how big are the chunks 

    Returns:
    ---------
    image: (time, dim3, dim4) where dim3 = dim1//grid_size + 1 etc
    """
    dim3 = range_size(image.shape[1], grid_size)
    dim4 = range_size(image.shape[2], grid_size)
    rois = np.empty((image.shape[0], dim3, dim4))
    for i, j in rois_indices(image.shape[1:], grid_size):
        big_i, big_j = i*grid_size, j*grid_size
        if i+grid_size < image.shape[1] and j + grid_size < image.shape[2]:
            roi = image[:, big_i:big_i+grid_size, big_j:big_j+grid_size]
        else:
            roi = image[:, big_i:, big_j:]
        signal = signal_from_roi(roi)
        rois[:, i, j] = signal
    return rois
        
def signal_from_roi(roi:np.ndarray, mode:str = 'mean'):
    """
    Takes the whole time roi and computes the signal given the function in mode

    Parameters:
    roi: (np.ndarray) (time, dim1, dim2)
    mode: (str) how to compute the signal in time
    """
    new_roi = roi.reshape((roi.shape[0], roi.shape[1] * roi.shape[2]))
    match mode:
        case 'median':
            func = np.median
        case 'mean':
            func = np.mean
        case _:
            raise NotImplementedError(F"Mode has to be 'mean' or 'meadian'. You used '{mode}'")
    roi_signal = func(new_roi, axis = 1)
    return roi_signal

def roi_activity(roi_signal:np.ndarray, scalar:int = 1):
    """
    Scalar: the threshold is computed as median + scalar*MAD

    It calculate threshold for each signal and then take median out of all
    these thresholds. It uses both of these to threshold the signal in order
    to get rid of low scale responses
    """
    thresholds = get_thresholds(roi_signal, scalar)
    threshold = np.median(thresholds)
    return (roi_signal > thresholds) * (roi_signal > threshold)

########################
# Signal processing
def baseline_als(y, lam, p, niter=10):
    """
    Gets the baseline of the signal with peaks
    """
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = sparse.linalg.spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """
    Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    https://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay

    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).

    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.

    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()

    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.asmatrix([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def image_filter(roi_signals, step:float = 1, kernel_size:int = 3):
    def for_one_signal(roi_signal:np.ndarray, step:float = 1, kernel_size:int = 3):
        minimum, maximum = np.min(roi_signal), np.max(roi_signal)
        line = np.arange(minimum, maximum, step = step)
        line = np.expand_dims(line, axis = 1)
        line = np.repeat(line, len(roi_signal), axis = 1)

        binary_image = line < np.expand_dims(roi_signal, axis = 0)

        kernel_shape = (kernel_size, kernel_size)
        kernel = np.ones(kernel_shape, np.uint8)

        opening = cv.morphologyEx(binary_image.astype(np.uint8), cv.MORPH_OPEN, 
                            kernel, iterations=1) 
        closing = cv.morphologyEx(opening.astype(np.uint8), cv.MORPH_CLOSE, 
                            kernel, iterations=1)
        signal = np.argmin(closing, axis = 0)
        base = baseline_als(signal, 10, 0.001)
        return signal -base
    
    func = lambda signal: for_one_signal(signal, step = step, kernel_size=kernel_size)
    return np.apply_along_axis(func, axis = 0, arr = roi_signals)

def filter_signals(signals:np.ndarray):
    filtered_func = lambda signal: savitzky_golay(signal, window_size=35, order = 4)
    filtered_signals = np.apply_along_axis(filtered_func, axis = 0, arr = signals)
    base_func = lambda signal: baseline_als(signal, lam = 10**2, p = 0.01)
    base = np.apply_along_axis(base_func, axis = 0, arr = filtered_signals)
    base[:15] = filtered_signals[:15]

    return filtered_signals - base


########################
# Creating output
def from_plt_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = image.crop((100, 100, 1100, 1100))

    return image

def create_video(filename:str, neurons:np.ndarray, neurons_active_rois:np.ndarray, astrocytes: np.ndarray, astrocytes_active_rois:np.ndarray, **kwargs):
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    video = cv.VideoWriter(filename, fourcc, 10, # frames per secod
                        (1000,1000) # The width and height come from the stats of image1
                        )
    
    rng = neurons.shape[0]
    # rng = 10 # I use this for testing
    for slice_index in tqdm.tqdm(range(rng),desc = 'Creating video',total = rng):
        fig, ax = plt.subplots(2,2, figsize = (12, 12))
        ax0 = plt.subplot(221)
        ax0.imshow(neurons[slice_index], cmap = 'grey')
        ax0.set_title('Neurons')
        ax0.set_axis_off()

        ax1 = plt.subplot(222)
        ax1.imshow(astrocytes[slice_index], cmap = 'grey')
        ax1.set_title('Astrocytes')
        ax1.set_axis_off()

        ax2 = plt.subplot(223)
        draw_active_rois(neurons[slice_index], neurons_active_rois[slice_index], ax = ax2, **kwargs)
        ax2.set_axis_off()

        ax3 = plt.subplot(224)
        draw_active_rois(astrocytes[slice_index], astrocytes_active_rois[slice_index], ax = ax3, **kwargs)
        ax3.set_axis_off()

        pil_image = from_plt_to_pil(fig)
        opencvImage = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)
        video.write(opencvImage)
        plt.close()
    video.release()

def active_rois_pipeline(time_sequence:np.ndarray, grid_size = DEFAULT_GRID_SIZE, mad_scalar:int = 1):
    # time_sequence = ndimage.gaussian_filter(time_sequence, sigma = 1, axes = (1,2))
    # norm_time_sequence = normalize(time_sequence, range_for_baseline=20)
    roi_signals = image_to_roi_signals(time_sequence, grid_size)
    # filtered_roi_signals = filter_signals(roi_signals)
    filtered_roi_signals = image_filter(roi_signals)
    active_rois = roi_activity(filtered_roi_signals, mad_scalar)
    return active_rois



def full_pipeline(file_path:str, grid_size:int = DEFAULT_GRID_SIZE, neurons_mad_scalar:int = 5, astrocytes_mad_scalar:int = 3,video:bool = False, video_name:str = None,full_path:bool = False, **kwargs):
    print("Loading data")
    neurons, astrocytes = get_data(file_path, full_path)
    print("Processing neurons")
    neurons_active_rois = active_rois_pipeline(neurons,grid_size = grid_size, mad_scalar=neurons_mad_scalar)
    # print(neurons_active_rois[:, 0, 0])
    print("Neurons done")
    print("Processing astrocytes")
    astrocytes_active_rois = active_rois_pipeline(astrocytes,grid_size = grid_size, mad_scalar = astrocytes_mad_scalar)
    print("Astrocytes done")
    
    kwargs['grid_size'] = grid_size
    if kwargs.get('color') is None:
        kwargs['color'] = 'r'
    if kwargs.get('linewidth') is None:
        kwargs['linewidth'] = 0.6   
        
    if video:
        if video_name is None:
            file, _ = os.path.splitext(os.path.basename(file_path))
            video_name = file + '.mp4'
            video_name = os.path.join(RESULTS_FOLDER, video_name)
        print(video_name)
        create_video(video_name, neurons, neurons_active_rois, astrocytes, astrocytes_active_rois, **kwargs)

    return neurons, neurons_active_rois, astrocytes, astrocytes_active_rois


########################
# Other fucntions
def n_active_rois(active_rois):
    return np.squeeze(np.apply_over_axes(np.sum, active_rois, [1,2]))

def shift_signal(signal:np.ndarray, shift:int):
    z = np.zeros((shift))
    new_signal = np.concatenate((signal, z))
    new_signal = np.roll(new_signal, shift)
    return new_signal[:-shift]


if __name__ == '__main__':
    full_pipeline('/home/s233039/Desktop/KU/data/5_70uA 10 trials 500ms.tiff')
