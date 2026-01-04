from .consts import DEFAULT_GRID_SIZE
from .input_output import get_data, image_to_roi_signals, current, is_ttx, experiment_number, load
from .filters import filter_signals, baseline_als, filter_lonely_spatial_activation, filter_lonely_temporal_activation
from .plotting import plot_roi_spatial_activity, plot_roi_temporal_activity
from .helpers import n_active_rois


import numpy as np

def MAD(a:np.ndarray, median:float = None):
    """
    Computes Median Absolute Deviation

    a: array
    median: in case it was already computed so we don't have to compute it again
    """
    if median is None:
        median = np.median(a, axis = 0)
    return np.median(np.abs(a - median), axis = 0)

# def get_thresholds(a:np.ndarray, scalar:int = 1, frames_before_first_impulse:int = 16) -> float:
#     """
#     a: array, signal
#     scalar: (int) How to scale the MAD   
#     frames_before_first_impulse: The median shouldn't include signal with responses to the impulses
#         because they then become included in normal behaviour 
#         if None, whole signal will be taken
#         Default is 16, as most of our experiments has that value
#     """
#     if frames_before_first_impulse is not None:
#         a = a[:frames_before_first_impulse, :, :]
#     median = np.median(a, axis = 0)
#     return median + scalar*MAD(a, median)

def get_thresholds(a:np.ndarray, scalar:int = 1) -> float:
    """
    a: array, signal
    scalar: (int) How to scale the MAD   
    """
    median = np.median(a, axis = 0)
    return median + scalar*MAD(a, median)

def roi_activity(roi_signal:np.ndarray, scalar:int = 1):
    """
    It calculate threshold for each signal and then take median out of all
    these thresholds. It uses both of these to threshold the signal in order
    to get rid of low scale responses

    Scalar: the threshold is computed as median + scalar*MAD
    """
    thresholds = get_thresholds(roi_signal, scalar)
    global_threshold = np.median(thresholds)
    return (roi_signal > thresholds) * (roi_signal > global_threshold)

def active_rois_pipeline(time_sequence:np.ndarray, 
                         grid_size = DEFAULT_GRID_SIZE, 
                         mad_scalar:int = 1):
    
    roi_signals = image_to_roi_signals(time_sequence, grid_size)
    filtered_roi_signals = filter_signals(roi_signals)
    active_rois = roi_activity(filtered_roi_signals, mad_scalar)
    
    # Get rid of signals that looks like random noise
    active_rois = filter_lonely_temporal_activation(active_rois)
    # active_rois = filter_lonely_spatial_activation(active_rois)

    return active_rois

def processing_pipeline(filename:str,
                        grid_size:int = DEFAULT_GRID_SIZE,
                        mad_scalar:int = 2
                        ):
    
    print("Loading data")
    neurons, astrocytes = get_data(filename)

    print("Processing neurons")
    neurons_active_rois = active_rois_pipeline(neurons,
                                               grid_size = grid_size,
                                                mad_scalar= mad_scalar)
    print("Neurons done")

    print("Processing astrocytes")
    astrocytes_active_rois = active_rois_pipeline(astrocytes,
                                                  grid_size = grid_size,
                                                  mad_scalar = mad_scalar)   
    print("Astrocytes done")
    
    return neurons, neurons_active_rois, astrocytes, astrocytes_active_rois


def plotting_pipeline(filename:str, 
                    #   neurons_active_rois:np.ndarray, 
                    #   astrocytes_active_rois:np.ndarray,
                      results_folder:str
                      ):
    
    # Plotting temporal activity
    _, neurons_active_rois, _, astrocytes_active_rois = processing_pipeline(filename)
    neurons_n_active_rois = n_active_rois(neurons_active_rois)
    astrocytes_n_active_rois = n_active_rois(astrocytes_active_rois)
    # neurons_n_active_rois = 5
    # astrocytes_n_active_rois = 4
    plot_roi_temporal_activity(
        neurons_n_active_rois, 
        astrocytes_n_active_rois,
        exp_info = (experiment_number(filename), current(filename), is_ttx(filename)),
        save_fig=True,
        results_folder = results_folder
        )
    plot_roi_spatial_activity(
        neurons_active_rois,
        astrocytes_active_rois, 
        exp_info= (experiment_number(filename), current(filename), is_ttx(filename)),
        results_folder = results_folder
    )
    return neurons_n_active_rois, astrocytes_n_active_rois


def preprocess(filename:str):
    info = load(filename)
