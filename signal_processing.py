import numpy as np
import filters
from consts import *


def normalize(rois:np.ndarray, first_stimul_index:int) -> np.ndarray:
    ss1 = first_stimul_index
    avg = np.average(rois[:ss1], axis = 0)
    std = np.std(rois[:ss1], axis = 0)
    return (rois - avg)/std

def get_baseline(signal:np.ndarray):
    return filters.baseline_als(signal, 30, 0.001)

def get_mva(signal:np.ndarray):
    return filters.moving_average(signal, 7)


def normalize_baseline_std(rois:np.ndarray, first_stimul_index:int) -> np.ndarray:
    """
    Normalizes the signal by substracting mean of signal before the first stimulation, 
    then divides by std of signal before the first stimulation.
    Filters by moving average with window size 7.
    Gets the baseline and substracts that.
    The other signal is without moving average filtering
    """

    ss1 = first_stimul_index

    def process_signal(signal, mva:bool):
        signal = (signal - np.average(signal[:ss1]))/np.std(signal[:ss1])
        if mva:
            signal = get_mva(signal)
        baseline = get_baseline(signal)
        return signal - baseline
    
    norm_filter = lambda signal: process_signal(signal, True)
    norm = lambda signal: process_signal(signal, False)
    
    norm_filtered_rois = np.apply_along_axis(norm_filter, axis = 0, arr = rois)
    normed_rois = np.apply_along_axis(norm, axis = 0, arr = rois)

    return norm_filtered_rois, normed_rois



def find_peaks(rois:np.ndarray, 
               stimulation_starts:np.ndarray,
               fs:float, 
               window_length:float,
               window_shift:float = 0):
    """
    Finds maximum value after stimulation within the first few seconds
    defined by window.
    window is expected in seconds
    Returns np.ndarray of shape (10, rois,shape[1], rois.shape[2])
    """
    ret = np.empty((10, rois.shape[1], rois.shape[2]))
    window_length = int(window_length*fs)
    window_shift = int(window_shift*fs)
    for index, start in enumerate(stimulation_starts):
        ret[index] = np.max(rois[start + window_shift:start + window_shift +window_length], axis = 0)

    return ret

def find_activity(
                rois:np.ndarray, 
               stimulations_starts:np.ndarray,
               fs:float, 
               window_length:float,
               window_shift:float =0,
               threshold:float = 2,
               time_above_threshold:float = None):
    """
    For each stimulation, looks for activity. It can be either just 
    surpassing the threshold or being above the threshold for some time.

    It can also look in a time window with delay (shift)

    """

    window_length = int(window_length*fs)
    window_shift = int(window_shift*fs)

    active_peaks = np.empty((10, rois.shape[1], rois.shape[2]))
    for index, start in enumerate(stimulations_starts):
        segment = rois[start + window_shift:start+ window_shift +window_length]
        segment_above_threshold = np.sum(segment > threshold, axis = 0)
        if time_above_threshold is not None:
            activity = segment_above_threshold > time_above_threshold*fs
        else:
            activity = segment_above_threshold > 0
        active_peaks[index] = activity
    return active_peaks

def response_delay_and_duration(
                filtered_rois:np.ndarray, 
                unfiltered_rois:np.ndarray,
               stimulations_starts:np.ndarray,
               fs:float, 
               window_length:float,
               window_shift:float =0,
               threshold:float = 2,
               time_above_threshold:float = None,
               ):
    
    delay = np.empty((10, filtered_rois.shape[1], filtered_rois.shape[2]))
    duration = np.empty((10, filtered_rois.shape[1], filtered_rois.shape[2]))

    window_length = int(window_length*fs)
    window_shift = int(window_shift*fs)

    for index, start in enumerate(stimulations_starts):
        unfiltered_segment = unfiltered_rois[start + window_shift:start + int(10*fs)]
        filtered_segment = filtered_rois[start + window_shift:start+ window_shift +window_length]

        filtered_segment_above_threshold = filtered_segment > threshold
        unfiltered_segment_above_threshold = unfiltered_segment > threshold
        peak_value = np.max(unfiltered_segment, axis = 0)
        unfiltered_segment_above_half_max = unfiltered_segment > peak_value/2
        
        unfiletered_duration_temp = np.sum(unfiltered_segment_above_threshold*unfiltered_segment_above_half_max, axis = 0).astype(np.float16)
        filetered_duration_temp = np.sum(filtered_segment_above_threshold, axis = 0).astype(np.float16)
        unfiletered_duration_temp[filetered_duration_temp < (time_above_threshold*fs)] = np.nan

        delay_temp = np.argmax(unfiltered_segment_above_threshold, axis = 0).astype(np.float16) + window_shift
        delay_temp[filetered_duration_temp < (time_above_threshold*fs)] = np.nan

        delay[index] = delay_temp/fs
        duration[index] = unfiletered_duration_temp/fs

    return delay, duration

    





def get_activity(info):
    neurons_activity = find_activity(info['precomputed']['precomputed_neurons'][0],
                                                info['stimulations_starts'],
                                                fs = info['fs'],
                                                window_length=NEURONS_WINDOW_LENGTH,
                                                time_above_threshold=NEURONS_TIME_ABOVE_THRESHOLD,
                                                window_shift=NEURONS_DELAY)

    astrocytes_activity = find_activity(info['precomputed']['precomputed_astrocytes'][0],
                                                info['stimulations_starts'],
                                                fs = info['fs'],
                                                window_length=ASTROCYTES_WINDOW_LENGTH,
                                                time_above_threshold=ASTROCYTES_TIME_ABOVE_THRESHOLD,
                                                window_shift=ASTROCYTES_DELAY)
    return neurons_activity, astrocytes_activity

def get_peaks(info):

    neurons_peaks =  find_peaks(info['precomputed']['precomputed_neurons'][1],
                                                info['stimulations_starts'],
                                                fs = info['fs'],
                                                window_length=NEURONS_WINDOW_LENGTH,
                                                window_shift=NEURONS_DELAY)


    astrocytes_peaks = find_peaks(info['precomputed']['precomputed_astrocytes'][1],
                                                info['stimulations_starts'],
                                                fs = info['fs'],
                                                window_length=ASTROCYTES_WINDOW_LENGTH,
                                                window_shift=ASTROCYTES_DELAY)
    return neurons_peaks, astrocytes_peaks