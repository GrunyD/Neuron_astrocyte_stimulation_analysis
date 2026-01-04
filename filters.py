from math import factorial

import numpy as np
from scipy import sparse, signal
from scipy.interpolate import UnivariateSpline

def bilateral_filter(signal:np.ndarray, 
                     window_size:int = 10, 
                     spatial_sigma:float = 5, 
                     edge_aware_sigma:float = 0.5) -> np.ndarray:
    """
    Low pass filter that preserves relatively strong changes in intensity
    Search for bilateral filter if in doubt

    Parameters:
    --------------
    - signal: 1D array to be filtered
    - window_size: over how many samples we do the concolution
    - spatial_sigma: std used for normal spatial gaus
    - edge_aware_sigma: std used for edge aware gaus
    """

    sig_min = np.min(signal)
    edit_signal = (signal - sig_min)
    sig_max = np.max(edit_signal)
    edit_signal = edit_signal/sig_max

    normal_dist = lambda x, sigma: np.exp(-(x**2)/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)
    spatial_gaus = np.arange(-window_size//2, window_size//2)
    spatial_gaus = normal_dist(spatial_gaus, spatial_sigma)

    new_signal = np.empty_like(signal)
    signal = np.pad(signal, window_size//2, mode = 'edge')

    for index in range(window_size//2, len(signal) - window_size//2):
        signal_slice = signal[index-window_size//2:index + window_size//2]
        edge_aware_input =  signal_slice - signal[index]
        edge_aware_gaus = normal_dist(edge_aware_input, edge_aware_sigma)
        normalization_const = np.sum(edge_aware_gaus*spatial_gaus)
        new_signal[index - window_size//2] = np.sum(spatial_gaus*edge_aware_gaus*signal_slice)/normalization_const

    new_signal = new_signal*sig_max # Scaling it back, because the abs values are important
    return new_signal

def baseline_als(y, lam:int, p:float, niter=10) -> np.ndarray:
    """
    Based on Assymetric least square smoothing.
    Taken from https://stackoverflow.com/questions/29156532/python-baseline-correction-library

    Parameters:
    -----------
    lam: The higher the more smooth is the baseline
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

def msbackadj(y, window_size=45, quantile=0.25, spline_smooth=1e-2):
    """
    Python equivalent of MATLAB's msbackadj function for baseline correction.
    
    Parameters:
        x (array-like): The x-axis values.
        y (array-like): The y-axis values (intensity values to correct).
        window_size (int): Size of the moving window for baseline estimation.
        quantile (float): Quantile level for baseline estimation (default: 0.1 for 10th percentile).
        spline_smooth (float): Smoothing factor for the spline interpolation.
    
    Returns:
        y_corrected (numpy array): The baseline-corrected y values.
    """
    y_baseline = np.zeros_like(y)
    half_window = window_size // 2
    
    # Moving window baseline estimation
    for i in range(len(y)):
        start = max(0, i - half_window)
        end = min(len(y), i + half_window)
        y_baseline[i] = np.quantile(y[start:end], quantile)  # Estimate baseline using quantile
    
    # Smooth the estimated baseline using a spline
    y_baseline_decimated = y_baseline[::10]
    x = np.arange(len(y))
    spline = UnivariateSpline(x[::10], y_baseline_decimated, s=spline_smooth)
    y_smooth_baseline = spline(x)
    # y_smooth_baseline = y_baseline
    # Subtract baseline from original signal
    # y_corrected = y - y_smooth_baseline
    
    return y_smooth_baseline

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

def get_baseline_and_filtered(signals:np.ndarray)->tuple[np.ndarray, np.ndarray]:
    """
    Applies signal filtering and calculates the baseline to substract. 
    Right now uses bilateral filter and msbackadj.

    Parameters:
        - signals: 3D numpy array, first axis is time

    Returns:
        - filtered signal
        - baseline to be substract from the signal
    """
    filtering_func = lambda signal: bilateral_filter(signal)
    filtered_signals = np.apply_along_axis(filtering_func, axis = 0, arr = signals)
    baseline_func = lambda filtered_signal: msbackadj(filtered_signal)
    baselines = np.apply_along_axis(baseline_func, axis = 0, arr = filtered_signals)

    return filtered_signals, baselines

def moving_average(signal:np.ndarray, window_length:int):
    return np.convolve(signal, np.full(window_length, 1/window_length))

def filter_signals(signals)->np.ndarray:
    f,b = get_baseline_and_filtered(signals)
    return f - b

def filter_lonely_temporal_activation(
        active_rois:np.ndarray, 
        least_neighbors:int = 1,
        neighborhood:np.ndarray = None) -> np.ndarray:
    """
    If the active roi is not active also before or after 
    then we could consider it random

    Parameters:
    --------------
    active_rois:
    least_neighbors: How many active ROIs have to be in neighborhood 
    neighborhood: where we look for active neighbors. 1D array is expected
        Should contain 1 in places where we expect activity and 0 where we
        don't care. The ROI in question is also 0. If None it defaults to
        (1, 0, 1)
    """
    if neighborhood is None:
        neighborhood = np.array((1,0,1))
    def filter_random_activation_inner(array:np.ndarray):
        """
        array: 1D array
        """
        return ((np.convolve(array, neighborhood, mode = 'same') >= least_neighbors) * array) > 0


    return np.apply_along_axis(filter_random_activation_inner, 
                               axis = 0, 
                               arr = active_rois
                               )

def filter_lonely_spatial_activation(
        active_rois:np.ndarray, 
        least_neigbors:int = 1,
        neighborhood:np.ndarray = None)->np.ndarray:
    """
    If active ROI is alone and has no neighbors, it is considered to be
     activationi by noise and it will be filtered out.
    
    Parameters:
    --------------
    - active_rois
    - least_neighbors: defines how many active neighbors the ROI must have
    - neighborhood: defines where we look for active neighbors. 2D array
        expected, with 1 at the places where we look for neighbors and 0
        everywhere else (including the middle). If None, it defaults to 
        the following shape:
        0 1 0
        1 0 1
        0 1 0
    """
    def filter_random_activation_inner(array:np.ndarray):
        """
        array: 2D array
        """
        if neighborhood is None:
            neighborhood = np.array(((0,1,0), (1,0,1), (0,1,0)))
        ret =  (signal.convolve2d(array, neighborhood, mode = 'same') >= least_neigbors) * array
        return ret > 0
    
    # TODO check if I can use apply along axis
    #Goes through slices in time
    for i in range(active_rois.shape[0]):
        active_rois[i] = filter_random_activation_inner(active_rois[i])
    return active_rois

