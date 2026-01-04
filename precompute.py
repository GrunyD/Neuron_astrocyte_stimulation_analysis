import os
import pickle

from .input_output import load, image_to_roi_signals
from .signal_processing import normalize_baseline_std

folder = ... # Folder containing data
viable_mice = (66, 72, 71, 78, 79, 81, 82)
for mouse_number in os.listdir(folder):
    if int(mouse_number) in viable_mice:
        exp_folder = os.path.join(folder, mouse_number, "microelectrode_experiments")
        for file in os.listdir(exp_folder):
            if file.endswith(".tiff"):
                filepath = os.path.join(exp_folder, file)
                d = load(filepath)
                grid_size = int(16/d['x']) # 15-16 microns per ROI, the resolutions vary slightly
                neurons_rois = image_to_roi_signals(d["neurons"], grid_size=grid_size)
                astrocytes_rois = image_to_roi_signals(d['astrocytes'], grid_size=grid_size)
                dict_to_pickle = {"precomputed_neurons": normalize_baseline_std(neurons_rois, d["stimulations_starts"][0]),
                                  "precomputed_astrocytes": normalize_baseline_std(astrocytes_rois, d["stimulations_starts"][0])}
                with open(filepath[:-5] + ' precomputed.pkl', "wb") as f:
                    pickle.dump(dict_to_pickle, f)

                print("Done: ", filepath)