import os
import json
import pickle

import numpy as np
import tifffile


DEFAULT_GRID_SIZE = 8
LOW_CURRENTS = [5, 10, 15, 20, 25, 30]
SHOWCASE_CURRENTS = [10, 20, 30, 50, 70,]
HIGH_CURRENTS = [50, 70, 100]
ALL_CURRENTS = [5, 10, 15, 20, 25, 30, 50, 70, 100]

EXAMPLE_FILE = '/home/s233039/Desktop/KU/data/79/microelectrode_experiments/5_70uA 10 trials.tiff'
DATA_FOLDER = "/home/s233039/Desktop/KU/data"

########################
# Functions for getting information from the filename
range_size = lambda shape, grid_size: shape//grid_size + int(shape%grid_size > 0)

file_by_number = lambda number, folder: [file for file in os.listdir(folder) if file.startswith(F'{str(number)}_')  and file.endswith('.tiff')][0]

is_ttx = lambda filename: 'ttx' in filename

exp_title = lambda number, current, ttx: F'Experiment number {number}: {current}uA with{" " if ttx else "out "}ttx'

def current(filename:str):
    """
    Gets the current used in an experiment based on the file name
    """
    filename = os.path.basename(filename)
    first_part = list(filename.split(' '))[0]
    second_part = list(first_part.split('_'))[-1]
    current = int(second_part[:-2])
    return current

def experiment_number(filename:str):
    """
    Gets the experiment number based on the filename
    """
    filename = os.path.basename(filename)
    first_part = list(filename.split(' '))[0]
    return int(list(first_part.split('_'))[0])

########################
# Loading data

def mouse_infos(
        data_folder:str = DATA_FOLDER, 
        load_ttx:bool = False,
        only_mouse:int = None,
        currents:str = 'showcase'
        ):
    
    if only_mouse is not None:
        viable_mice = (only_mouse, )
    else:
        viable_mice = (71, 72, 78, 79, 80)

    if currents == 'showcase':
        currents = SHOWCASE_CURRENTS
    elif currents == 'low':
        currents = LOW_CURRENTS
    elif currents == 'high':
        currents = HIGH_CURRENTS
    elif currents == 'all':
        currents = ALL_CURRENTS
    else:
        raise ValueError(f'Currents can be either "showcase", "all", "low" or "high", not {currents}')
    
    for mouse_number in os.listdir(data_folder):
        if int(mouse_number) in viable_mice:
            exp_folder = os.path.join(data_folder, mouse_number, "microelectrode_experiments")
            for file in os.listdir(exp_folder):
                if any(list(map(lambda x: str(x)+"uA" in file, currents))):
                    if not load_ttx and 'ttx' in file:
                        continue
                    if 'baseline' in file:
                        continue
                    if file.endswith(".tiff"):
                        filepath = os.path.join(exp_folder, file)
                        info = load(filepath,  load_ttx = True)
                        yield info

def load(path:str, verbose:bool = False, load_ttx:bool = False) -> dict:
    """
    Loads the sequence and important additional data like sampling
    frequency and pixel resolution
    """
    filename = os.path.basename(path)
    mouse_dir = os.path.dirname(os.path.dirname(path))
    if verbose:
        print(f"Mouse {mouse_dir}, {filename} - ")
    info_file = os.path.join(mouse_dir, f"experiments_info/{filename[:-4]}txt")
    neurons, astrocytes = get_data(path, verbose)
    info = get_info(info_file)
    info["mouse_number"] = int(mouse_dir[-2:])
    info["exp_number"] = experiment_number(filename)
    info["current"] = current(filename)
    info["neurons"] = neurons
    info["astrocytes"] = astrocytes
    info["stimulations_starts"] = stimulations_starts(info["fs"])
    if not load_ttx:
        info["baselines"] = tifffile.imread(mouse_dir + '/microelectrode_experiments/' + f"{info['exp_number']}_{info['current']}uA baselines.tiff")
    # info["tip_location"] = get_tip_location(mouse_dir)
    try:
        info['precomputed'] = get_precomputed_data(path[:-5] + ' precomputed.pkl')
    except FileNotFoundError:
        print("Doesn't have precomputed file")
    info['is_ttx'] = "ttx" in filename

    return info
    

def get_data(filename:str, verbose:bool = False) -> tuple:
    """
    Loads data and returns two time sequences. 
    Expects that the channel axis is axis number 2

    Parameters:
    -----------
    - filename: name of the file in the data_folder

    Retruns:
    -----------
        (Channel 1, Channel 2, both numpy arrays)
        (Neurons, Astrocytes)
    """ 
    image = tifffile.imread(filename)
    if verbose:
        print("Data shape: ", image.shape)
    return image[:, 0], image[:, 1]

def get_precomputed_data(filename:str) -> dict:
    """
    For efficiency of plotting, the operation of filtering and
    baseline leveling was precomputed and saved as pickled objects.

    It contains a dictionary with 'precomputed_neurons' and 'precomputed_astrocytes'

    These are the direct outputs from signal_processing.normalizes_baseline_std
    """
    with open(filename, "rb") as f:
        d = pickle.load(f)
    return d    


def get_info(filename) -> dict:
    ret_dict = {}
    keys = ["x", "y", "fs"]
    funcs = [get_pixel_resolution, get_pixel_resolution, get_fs]
    line_indices = [9, 10, 12]
    with open(filename, "r") as f:
        lines = f.readlines()
    for index, func, key in zip(line_indices, funcs, keys):
        ret_dict[key] = func(lines[index])
    ret_dict['grid_size'] = int(16/ret_dict['x'])

    return ret_dict

def get_pixel_resolution(line:str) -> float:
    """
    Returns resolution in um/pixel
    """
    resolution = float(line.split(",")[2].strip().split(" ")[0])
    return resolution

def get_fs(line:str)  -> float:
    """
    Loads the sampling frequency of that experiment from the corresponding
    textfile
    """
    
    line = line[15:-23].split(",")
    n_slices = int(line[0])
    times = list(map(float, line[1].split("-")))
    time = times[1] - times[0]
    return n_slices/time

def get_tip_location(dir) -> list:
    with open(os.path.join(dir, "microelectrode_location/info.json"), "r") as f:
        d = json.load(f)
    return d["tip_location"]

def stimulations_starts(fs:float) -> np.ndarray:
    return np.array([int(fs*event*10.5)-2 for event in range(1, 11)])

########################
# Iterator through indices based on grid size
def rois_indices(image_shape:(tuple), grid_size:int = DEFAULT_GRID_SIZE):
    for i in range(range_size(image_shape[0], grid_size)):
        for j in range(range_size(image_shape[1], grid_size)):
            yield i, j

# def normalize(a:np.ndarray, mode:str = 'median', range_for_baseline:int = None):
#     """
#     Returns (a-F0)/F0
#     computes F0 across axis = 0

#     Parameters:
#     ------------
#     a: (np.ndarray) array to be normalized accros frames
#     mode: (str) how to compute F0. Either 'mean' or 'median'
#     range_for_baseline: (int) in case the baseline is not computed over the whole signal
#         but only the first part, this defines (in frames) how long is the first part
#     """
#     match mode:
#         case 'median':
#             func = np.median
#         case 'mean':
#             func = np.mean
#         case _:
#             raise NotImplementedError(F"Mode has to be 'mean' or 'meadian'. You used '{mode}'")
    
#     if range_for_baseline is not None:
#         F0 = func(a[:range_for_baseline], axis = 0) 
#     else:
#         F0 = func(a, axis = 0)
    
#     return (a-F0)/F0

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