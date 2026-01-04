from .consts import DEFAULT_GRID_SIZE
from .plotting import draw_active_rois, from_plt_to_pil

import os

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from tqdm import tqdm

def create_video(filename:str, 
                 neurons:np.ndarray, 
                 neurons_active_rois:np.ndarray, 
                 astrocytes: np.ndarray, 
                 astrocytes_active_rois:np.ndarray,
                 results_folder:str,
                 grid_size:int = DEFAULT_GRID_SIZE,
                 **kwargs):
    
    file, _ = os.path.splitext(os.path.basename(filename))
    video_name = file + '.mp4'
    video_name = os.path.join(results_folder, video_name)

    kwargs['grid_size'] = grid_size
    if kwargs.get('color') is None:
        kwargs['color'] = 'r'
    if kwargs.get('linewidth') is None:
        kwargs['linewidth'] = 0.6 


    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    video = cv.VideoWriter(video_name, fourcc, 10, # frames per secod
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