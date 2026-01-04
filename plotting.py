from consts import DEFAULT_GRID_SIZE
from input_output import rois_indices, exp_title, image_to_roi_signals
from helpers import (get_events, 
                      n_active_rois, 
                      tip_location_roi, 
                      dice_score, 
                    #   get_image,
                      overlap_image_map)
from filters import baseline_als
from signal_processing import normalize_baseline_std, find_peaks, find_activity
from image_processing import (
                                enhance_contrast, 
                               get_image, 
                               segment_neurons, 
                               segment_astrocytes,
                               rois_from_segmentation,

                               )

import os
import io

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import PIL
import scipy

DEFAULT_LINE_WIDTH = 0.7
DEFAULT_CMAP = 'plasma'
SAVING_FOLDER = "/home/s233039/Desktop/KU/plots/"

def add_tip_location(ax, tip_xy):
    """
    ax is the one subplot where we want add tip_location
    """
    ax.add_patch(ptch.Rectangle(tip_xy, 1, 1, lw = 1, fc = 'none', ec = 'g'))

def imshow(fig, ax, image, vmin:float = None,cmap:str = 'plasma', vmax:float = None, title:str = None, colorbar:bool = True, alpha:float = 1):
    im = ax.imshow(image, vmin = vmin, vmax = vmax, alpha = alpha, cmap = cmap)
    if colorbar:
        fig.colorbar(im)
    if title is not None:
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
#####################################################################
###########             Decorator            ########################

def prepare_fig(info:dict ,rows:int, columns:int, figsize:tuple):
    fig, ax = plt.subplots(rows, columns,figsize = figsize, layout = 'compressed')
    title = f'Mouse {info["mouse_number"]}\nExp {info["exp_number"]}: {info["current"]} uA'
    fig.suptitle(title)
    return fig, ax

def remove_ticks(ax):
    """
    ax are all subplots in the figure
    """
    for im in ax.flatten():
        im.set_xticks([])
        im.set_yticks([])


def save_fig(fig, figure, name):
    fig.savefig(os.path.join(SAVING_FOLDER, f"Figure {figure}/{name}.svg"), format = "svg")


def create_and_save_figure(rows:int, columns:int, folder:str):
    def decorator(func):
        def wrapper(info):
            fig, ax = prepare_fig(info, rows, columns, (24, 13))
            plt.set_cmap(DEFAULT_CMAP)
            neurons_rois = image_to_roi_signals(info['neurons'])
            astrocytes_rois = image_to_roi_signals(info['astrocytes'])
            func(info, neurons_rois, astrocytes_rois, fig, ax)
            # remove_ticks(ax)
            save_fig(fig, info, folder)
        return wrapper
    return decorator    

####################################################################

def draw_full_grid(image:np.ndarray, grid_size:int = DEFAULT_GRID_SIZE, linewidth:float = 0.2, color:str = 'white', cmap = 'grey'):
    """
    Plots an image with full grid
    For testing purposes only
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
    ax.set_axis_off()
    for i, row in enumerate(active_rois):
        for j, is_active in enumerate(row):
            if bool(is_active):
                rect = ptch.Rectangle((j*grid_size, i*grid_size), grid_size, grid_size, edgecolor = color, facecolor = 'none', linewidth = linewidth)
                ax.add_patch(rect)
    if ax is None:
        plt.show()

######################################################################
# For spider plots


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.

    """
    from matplotlib.patches import Circle, RegularPolygon
    from matplotlib.path import Path
    from matplotlib.projections import register_projection
    from matplotlib.projections.polar import PolarAxes
    from matplotlib.spines import Spine
    from matplotlib.transforms import Affine2D
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta




#########################################################################š

def plot_roi_spatial_activity(active_rois_n:np.ndarray,active_rois_a:np.ndarray, exp_info:tuple, results_folder:str):
    """
    Saves 7 plots for the experiment:
        1) How much of total time were neuronal rois active

        (only astrocytes from now on)
        2) How much of total time were astrocytal rois active
        3) How many events rois responded to (event is from start to the end of excitation)
        4) How long were rois active during an event (event is from start to the end of excitation)
        5) How many events rois responded to (event is from start of excitation to the start of the next excitation)
        6) How long were rois active during an event (event is from start of excitation to the start of the next excitation)
        7) Average delay in response to neuronal excitation
    """
    def imshow(image, title, filename, color_bar_label:str = None):
        fig, ax = plt.subplots()
        axes_image = ax.imshow(image)
        ax.set_title(F'{title}\n{exp_title(*exp_info)}')
        if color_bar_label is not None:
            fig.colorbar(axes_image, label = color_bar_label)
        else:
            fig.colorbar(axes_image)
        ax.set_axis_off()
        fig.savefig(os.path.join(results_folder, F'{exp_info[0]}_{filename}'))
        plt.close()

    def scatter(x, y, title, filename):
        fig, ax = plt.subplots()
        axes_plot = ax.scatter(x, y)

        ax.set_title(F'{title}\n{exp_title(*exp_info)}')
        fig.savefig(os.path.join(results_folder, F'{exp_info[0]}_{filename}'))
        plt.close()

    def relative_to_time_activity(active_rois:np.ndarray, cell_type:str):
        image = np.sum(active_rois, axis = 0)/active_rois.shape[0]
        title = F'Relative RoI activity of {cell_type}'
        filename = F'roi_spatial_activity_{cell_type}.png'
        imshow(image, title, filename)

    relative_to_time_activity(active_rois_n, 'neurons')
    relative_to_time_activity(active_rois_a, 'astrocytes')

    n_active_rois_neurons = n_active_rois(active_rois_n)
    events_threshold = np.max(n_active_rois_neurons)/2
    events = get_events(n_active_rois_neurons>events_threshold)

    # Event starts with exceeding threshold and ends with going below thgreshold
    heatmap1 = np.zeros_like(active_rois_a[0], dtype = int) # How many events roi responded to
    heatmap2 = np.zeros_like(active_rois_a[0], dtype=int) # How long was roi active during events in total
    
    # Event starts with exceeding threshold and ends with start with next event
    heatmap3 = np.zeros_like(active_rois_a[0], dtype = int) # How many events roi responded to
    heatmap4 = np.zeros_like(active_rois_a[0], dtype = int) # How long was roi active during events in total
    
    heatmap5 = np.zeros_like(active_rois_a[0], dtype = int) # Average time it takes for roi of astrocyte to get active after the start of an event

    for index, (start, end) in enumerate(events):
        s = np.sum(active_rois_a[start:end], axis = 0)
        heatmap1 += s > 0
        heatmap2 += s

        if index < len(events) - 1:
            s = np.sum(active_rois_a[start:events[index+1][1]], axis = 0)
        else:
            s = np.sum(active_rois_a[start:end], axis = 0)
        heatmap3 += s >0
        heatmap4 += s 

        started = np.zeros_like(active_rois_a[0])
        for i in range(start, end):
            # We want to check even behind the event
            if index < len(events) - 1:
                end = events[index+1][1]

            slice = active_rois_a[i]
            slice = slice * np.invert(started)
            heatmap5 += slice * (i - start)
            started += slice

    # Averaging over number of events it reacted to
    heatmap5 = heatmap5/heatmap3 
    heatmap5[heatmap3 < 4] = np.nan
    # Average over the number of events
    heatmap2 = heatmap2/len(events)
    heatmap4 = heatmap4/len(events)


    
    # imshow(heatmap1,
    #     title = 'Number of events that astrocytes reacted to',
    #     filename = 'roi_spatial_activity_during_events.png'
    # )

    # imshow(heatmap2*0.5, # Multiplying by sampling frequency
    #     title = 'Average lenght of ROI reaction during an event',
    #     filename = 'roi_spatial_activity_during_events-lenght_of_activity.png',
    #     color_bar_label = 'Time (s)'
    # )
    
    # imshow(heatmap3,
    #     title = 'Number of events that astrocytes reacted to',
    #     filename = 'roi_spatial_activity_during_events2.png'
    # )
    # imshow(heatmap4*0.5, # Multiplying by sampling frequency
    #     title = 'Average lenght of ROI reaction during an event',
    #     filename = 'roi_spatial_activity_during_events-lenght_of_activity2.png',
    #     color_bar_label = 'Time (s)'
    # )
    imshow(heatmap5*0.5, # Multiplying by sampling frequency
        title = 'Average delay in astrocyte response',
        filename = 'average delay in astrocyte response.png',
        color_bar_label = 'Time (s)'
    )
    #TODO load this for each FOW
    tip_location = [25, 4]
    delay_x, delay_y = astrocytes_delay_to_1D(heatmap5, tip_location)
    scatter(delay_x, 
            delay_y,
            title = 'Average delay in astrocyte response',
            filename = 'average delay in astrocyte response1D.png'
            )
    
def plot_roi_temporal_activity(roi_activity_n:np.ndarray, 
                               roi_activity_a:np.ndarray, 
                               exp_info:tuple, 
                               results_folder:str, 
                               grid_size:int = DEFAULT_GRID_SIZE, 
                               sample_freq:float = 2, 
                               save_fig:str = False):
    """
    roi_activity_n: How many ROIs were active at given time point for neurons
    roi_activity_a: How many ROIs were active at gievn time point for astrocytes
    exp_info: tuple with information about experiment, 
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
    ax.legend(['Neurons', 'Astrocytes'], loc = 'upper right')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(F'Number of active RoIs\ngrid size: {grid_size}')
    ax.set_title(F'Active RoIs in time\n{exp_title(*exp_info) if exp_info is not None else ""}')
    if save_fig:
        fig.savefig(os.path.join(results_folder, F'{exp_info[0]}_roi_temporal_activity.png'))
        plt.close()
    else:
        plt.show()

def plot_average_baseline(baselines:np.ndarray, 
                          mask:np.ndarray = None,
                          range:str = 'std',
                          save_fig:str = False):
    
    """
    Create plot of average baseline.

    Parameters:
        - Baselines
        - Mask: what rois we include. Expects 2D mask
        - range: How to show the range around average. Possibilities are
            'std' (default), 'range' (max and min), None (doesn't show)
        - save_fig: If it evaluates to True, it expects a path where to save
    """
    if mask is not None:
        baselines = baselines[mask]
    avg = np.average(baselines, axis = (1,2))

    fig, ax = plt.subplots()
    ax.plot(avg)

    if range is None:
        pass
    elif range == 'std':
        std = np.std()

def astrocytes_delay_to_1D(delay_heatmap:np.ndarray, 
                          tip_location:list):
    xi = np.arange(delay_heatmap.shape[0])[:, None]
    xi = np.broadcast_to(xi, (delay_heatmap.shape))
    xj = np.arange(delay_heatmap.shape[1])[None, ...]
    xj = np.broadcast_to(xj, delay_heatmap.shape)
    xi = xi - tip_location[0]
    xj = xj - tip_location[1]
    #TODO to real distance
    x = np.sqrt(xi**2 + xj**2)
    x = x.reshape(-1)
    delay = delay_heatmap.reshape(-1)
    index = ~np.isnan(delay)
    return x[index], delay[index]

def from_plt_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = image.crop((100, 100, 1100, 1100))

    return image


def plot_increased_baseline_heatmaps(
        neuron_rois:np.ndarray, 
        astrocyte_rois:np.ndarray, 
        info:dict):
    """
    Plots heatmaps which shows how much the baseline increases compared
    to first 10 seconds of the recording
    
    """
    ss = info["stimulations_starts"]
    event_time = np.median(np.diff(ss))
    np.append(ss, ss[-1]+event_time)

    def baseline_increase_heatmaps(signal):
        als_baseline = baseline_als(signal, 30, 0.001)
        values_at_starts = als_baseline[ss[:-1]]
        first_10s_baseline = np.median(signal[:ss[0]])

        avg_baseline_increase = np.average(values_at_starts[1:]/first_10s_baseline)
        start_baseline_increase = np.average(values_at_starts[1:4]/first_10s_baseline)
        return (
            avg_baseline_increase*100,
            start_baseline_increase*100
        )

    neurons_heatmaps = np.apply_along_axis(baseline_increase_heatmaps, axis = 0, arr = neuron_rois)
    astrocytes_heatmaps = np.apply_along_axis(baseline_increase_heatmaps, axis = 0, arr = astrocyte_rois)


    tip_xy = tip_location_roi(info["tip_location"], info['grid_size'])
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(10, 9)
    mouse_number = info["mouse_number"]
    exp_number = info['exp_number']
    current = info["current"]
    title = f"Mouse {mouse_number}\nexp {exp_number}: {current}uA"
    fig.suptitle(f"{title}\n\n", fontsize="x-large")

    plot_type = [
        "Average relative baseline increase in %",
        "Average realtive increase of\nbeginning of baseline in %",
    ]
    cell_type = ('Neurons', 'Astrocytes')
    for row, heatmaps in enumerate((neurons_heatmaps, astrocytes_heatmaps)):
        
        for col in range(len(heatmaps)):
            if row == 0:
                ax[0, col].set_title(plot_type[col])
            im = ax[row, col].imshow(heatmaps[col], cmap = 'plasma')
            fig.colorbar(im)
            ax[row, col].set_xticks([])
            ax[row, col].set_yticks([])
            if col == 0:
                ax[row, col].set_ylabel(cell_type[row], size = 'large')
            ax[row, col].add_patch(ptch.Rectangle(tip_xy, 1, 1, lw = 1, fc = 'none', ec = 'r'))

    fig.tight_layout()
    fig.savefig(f'/home/s233039/Desktop/KU/results/heatmaps/{mouse_number}/{exp_number}_{current}_uA baseline heatmap opt-colorbar.png')


def plot_peak_increase_heatmaps(
        neuron_rois:np.ndarray, 
        astrocyte_rois:np.ndarray, 
        info:dict
        ):
    """
    
    """
    ss = info["stimulations_starts"]
    event_time = np.median(np.diff(ss))
    np.append(ss, ss[-1]+event_time)

    def peak_increase_heatmaps(signal):
        als_baseline = baseline_als(signal, 30, 0.001)
        bases = als_baseline[ss[:-1]]
        split = np.split(signal, ss)[1:-1]
        peaks = np.array(list(map(np.max, split)))
        # bases = np.array([event[0] for event in split])
        # There is a big jump in the first stimulation, that shouldnt be included
        return (
            np.average(peaks[1:]/bases[1:])*100, 
            np.average(peaks[1:] - bases[1:]), 
            np.std(peaks[1:] - bases[1:]),
            np.average(peaks),
            np.std(peaks))

    neuron_heatmaps = np.apply_along_axis(peak_increase_heatmaps,0, neuron_rois)
    astrocyte_heatmaps = np.apply_along_axis(peak_increase_heatmaps,0, astrocyte_rois)

    fig, ax = plt.subplots(5, 2)
    fig.set_size_inches(10, 20)
    mouse_number = info["mouse_number"]
    exp_number = info['exp_number']
    current = info["current"]
    title = f"Mouse {mouse_number}\nexp {exp_number}: {current}uA"
    fig.suptitle(f"{title}\n\n", fontsize="x-large")
    ylabels = [
        "Average relative peak intensity increase in %",
        "Average raw peak intesity\ncompared to event start",
        "Standard deviation raw peak intensity\ncompared to event start",
        "Average absolute raw peak intensity",
        "Standard deviation of absolute\nraw peak intensity"
    ]
    titles = ('Neurons', 'Astrocytes')
    tip_xy = tip_location_roi(info["tip_location"], info['grid_size'])
    for j, heatmaps in enumerate((neuron_heatmaps, astrocyte_heatmaps)):
        ax[0, j].set_title(titles[j])
        for i in range(len(heatmaps)):
            im = ax[i, j].imshow(heatmaps[i])
            fig.colorbar(im)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            if j == 0:
                ax[i, j].set_ylabel(ylabels[i], size = 'large')
            ax[i, j].add_patch(ptch.Rectangle(tip_xy, 1, 1, lw = 1, fc = 'none', ec = 'r'))

    fig.tight_layout()
    print("edit")
    fig.savefig(f'/home/s233039/Desktop/KU/results/heatmaps/{mouse_number}/{exp_number}_{current}_uA peaks heatmap2.png')

    
def plot(info:dict):
    ss = info["stimulations_starts"]
    event_time = np.median(np.diff(ss))
    new_ss = np.roll(ss, -1)
    new_ss[-1] = new_ss[-2] + event_time
    def get_heatmaps(signal):
        als_baseline = baseline_als(signal, 30, 0.001)
        values_at_starts = als_baseline[ss[:-1]]
        first_10s = signal[:ss[0]]
        first_10s_baseline = np.median(first_10s)

        event_time = np.median(np.diff(ss))
        np.append(ss, ss[-1]+event_time)
        split = np.split(signal, ss)[1:-1]
        peaks = np.array(list(map(np.max, split)))
        bases = als_baseline[ss[:-1]]
        # bases = np.array([event[0] for event in split])

        avg_baseline_increase = np.average(values_at_starts[1:]/first_10s_baseline)
        start_baseline_increase = np.average(values_at_starts[1:4]/first_10s_baseline)
        relative_peak_increase = np.average(peaks/bases)
        relative_peak_increase_to_baseline_noise = np.average((peaks[1:] - bases[1:])/(np.max(first_10s) - np.min(first_10s)))
        relative_peak_increase_std = np.std(peaks/bases)

        lower_clip_100 = lambda arr: np.clip(arr, a_min = 100,a_max = None)
        return (
            lower_clip_100(avg_baseline_increase*100),
            lower_clip_100(start_baseline_increase*100),
            lower_clip_100(relative_peak_increase*100),
            relative_peak_increase_to_baseline_noise,
            relative_peak_increase_std*100
        )
    
    def enhance_contrast(image):
        image -= np.min(image)
        image = np.array(255*image/np.max(image), dtype = np.uint8)
        hist,_ = np.histogram(image, bins = 256, range = (0, 255))
        cumsum = np.cumsum(hist[::-1])
        maximum = np.sum(cumsum < np.size(image)/70)
        image = image/(255 - maximum)
        np.clip(image,a_min = None, a_max = 1, out = image)
        
        return image

    fig, ax = plt.subplots(4, 4,figsize = (15, 9), layout = 'compressed')
    title = f'Mouse {info["mouse_number"]}\nExp {info["exp_number"]}: {info["current"]} uA'
    fig.suptitle(title)
    
    def row_plot(row, sequence:np.ndarray, cell_type:str):

        # Displaying the image with correct channel color
        rgb_image = np.zeros((sequence.shape[1], sequence.shape[2], 3))
        image = np.average(sequence, 0)
        image = enhance_contrast(image)
        index = 0 if cell_type == 'Neurons' else 1
        rgb_image[:,:, index] = image
        
        ax[row, 0].imshow(rgb_image) 
        ax[row, 0].set_ylabel(cell_type)
        
        rois = image_to_roi_signals(sequence)
        rois_avg = np.average(rois, 0)
        rois_avg = enhance_contrast(rois_avg)
        # ax[row, 1].imshow(np.average(rois, 0), cmap = 'plasma')
        ax[row, 1].imshow(rois_avg, cmap = 'plasma')
        

        heatmaps = np.apply_along_axis(get_heatmaps, axis = 0, arr = rois)
        for i, (row_var, col) in zip(range(5), ((row, 2), (row, 3), (row+1, 1), (row+1, 2), (row+1, 3))):
            img = ax[row_var, col].imshow(heatmaps[i], cmap = 'plasma')
            fig.colorbar(img)
            dice = dice_score(rois_avg, heatmaps[i])
            # dice, _, _ = cv.EMD(rois_avg.astype(np.float32),heatmaps[i].astype(np.float32),cv.DIST_L1)
            ax[row_var, col].set_ylabel(f'Dice: {dice:.3f}')
            

        titles = [
            'Average over time',
            'Average over time RoIs',
            'Relative baseline increase in %',
            'Relative baseline increaase in %\n(First four pulses)',
            '',
            'Relative peak increase in %',
            'Relative peak increase to\nbaseline noise ration',
            'Relative peak increase std\nin % points'

        ]
        tip_xy = tip_location_roi(info["tip_location"], info['grid_size'])
        for i in range(8):
            ax[row + i//4, i%4].set_yticks([])
            ax[row + i//4, i%4].set_xticks([])
            ax[row + i//4, i%4].set_title(titles[i], size = 'small')
            if i > 0:
                ax[row+i//4, i%4].add_patch(ptch.Rectangle(tip_xy, 1, 1, lw = 1, fc = 'none', ec = 'g'))

        # Deletes the boarders of the white space plot
        ax[row+1, 0].set_axis_off()
        return rois, heatmaps
                
    neurons = info['neurons']
    neurons_rois, neurons_heatmaps = row_plot(0, neurons, 'Neurons')
    astrocytes = info['astrocytes']
    astrocytes_rois, astrocytes_heatmaps = row_plot(2, astrocytes, 'Astrocytes')

    # dice = 0
    # dice2 = 0
    # for i in range(30):
    #     r = np.random.random((32, 32))
    #     dice += nasa.helpers.dice_score(neurons_rois, r)
    #     dice2 += nasa.helpers.dice_score(astrocytes_rois, r)
    # print(dice/30, dice2/30)

    text = "Astrocytes-Neurons Dice score:\n"
    dice = dice_score(neurons_rois, astrocytes_rois)
    text = text + f"Intensity: {dice:.3f}"
    for i, t in zip(range(5), ("Baseline", "Baseline start", "Peaks", "Noise", "STD")):
        dice = dice_score(neurons_heatmaps[i], astrocytes_heatmaps[i])
        # dice, _, _ = cv.EMD(neurons_heatmaps[i].astype(np.float32),astrocytes_heatmaps[i].astype(np.float32),cv.DIST_L1)
        text = text + f"\n{t}: {dice:.3f}"
    ax[3, 0].text(0, 0, text + "\n\n\n\n", size = "small")
    mouse_number = info["mouse_number"]
    exp_number = info["exp_number"]
    current = info["current"]
    fig.savefig(f'/home/s233039/Desktop/KU/results/correlations/{mouse_number}/{exp_number}_{current}_uA.png')


def correlations(info):
    """
    magenta         red         orange
    darkblue       white        yellow
    lightblue      cyan         green
    """
    colors = np.array([
        [255, 0, 255], #magenta
        [255, 0, 0], # red
        [255, 127, 0], #orange
        [0, 0, 255], #darkblue
        [255, 255, 255], # white
        [255, 255, 0], #yellow
        [0, 127, 255], #lightblue
        [0, 255, 255], #cyan
        [0, 255, 0], # green
    ])/255

    def correlate(arr1, arr2):
        t = arr1.shape[0]
        def func(arr):
            corr = scipy.stats.pearsonr(arr[:t], arr[t:])
            return corr.statistic, corr.pvalue
        return np.apply_along_axis(func, axis = 0, arr = np.vstack((arr1, arr2)))

    def moving_correlation(non_padded, to_be_padded, padd):
        t = non_padded.shape[0]
        corrs = np.empty(((2*padd+1)**2, non_padded.shape[1], non_padded.shape[2]))
        padded = np.pad(to_be_padded, padd)[padd:-padd]
        for i in range(padd*2+1):
            end_i = -(2*padd-i) if i < 2*padd else None
            for j in range(padd*2+1):
                end_j = -(2*padd-j) if j < 2*padd else None

                func = lambda arr: scipy.stats.pearsonr(arr[:t], arr[t:]).statistic

                corrs[i*(padd*2+1) + j] = np.apply_along_axis(func, axis = 0, arr = np.vstack((non_padded, padded[:, i:end_i, j:end_j])))

        return corrs

    def distance(arr1, arr2, padd):
        corrs = moving_correlation(arr1, arr2, padd)
        corrs[np.isnan(corrs)] = -1
        arg = np.argmax(corrs, 0)
        m = np.max(corrs, 0)
        j, i = ((arg%(2*padd+1))-padd), -((arg//(2*padd+1))-padd)
        distance = np.sqrt(i**2 + j**2)
        a = np.arctan2(i, j)
        a_ret = np.ones_like(a, dtype = int)*3
        t = 2*np.pi/8
        thresholds = np.arange(8)*t - np.pi + np.pi/8
        c = [6, 7, 8, 5, 2, 1, 0]
        for index in range(7):
            n = (a>= thresholds[index]) * (a<thresholds[index+1])
            # i = i>0
            a_ret[n] = c[index]
        a_ret[(i == 0)*(j == 0)] = 4
        return distance, colors[a_ret.flatten()].reshape((a_ret.shape[0], a_ret.shape[1], 3)), m

    def colormap(arr1, arr2):
        corrs = moving_correlation(arr1, arr2, 1)
        corrs[np.isnan(corrs)] = -1
        arg = np.argmax(corrs, 0)
        heatmap = colors[arg.flatten()].reshape((arg.shape[0], arg.shape[1], 3))
        m = np.max(corrs, 0)
        return heatmap, m
    
    neurons_rois = image_to_roi_signals(info['neurons'])
    astrocytes_rois = image_to_roi_signals(info['astrocytes'])
    fig, ax = plt.subplots(2, 7,figsize = (15, 9), layout = 'compressed')
    title = f'Mouse {info["mouse_number"]}\nExp {info["exp_number"]}: {info["current"]} uA'
    fig.suptitle(title)

    rgb_image = get_image(info['neurons'], "Neurons")

    dist1,a1, m1 = distance(neurons_rois, astrocytes_rois, 5)
    dist2, a2, m2 = distance(astrocytes_rois, neurons_rois, 5)

    ax[0, 0].imshow(rgb_image) 
    ax[0, 0].set_title("Neurons")


    corrs = correlate(neurons_rois, astrocytes_rois)
    ax[0, 1].set_title("Correlation coefficient")
    im = ax[0, 1].imshow(corrs[0], cmap = 'plasma')
    fig.colorbar(im)
    ax[0,2].set_title("P value")
    im = ax[0, 2].imshow(corrs[1], cmap = 'plasma')
    fig.colorbar(im)
    ax[0,3].set_title("Direction colorcoding")
    ax[0, 3].imshow(colors.reshape((3,3, 3)))

    heatmap, scale = colormap(neurons_rois, astrocytes_rois)
    ax[0,4].set_title("Neighbor with\nhighest correlation")
    ax[0, 4].imshow(heatmap)

    ax[0,5].set_title("Highest correlation\nin neighborhood")
    im  = ax[0,5].imshow(scale, cmap = 'plasma')
    fig.colorbar(im)

    rgb_image = get_image(info['astrocytes'], "Astrocytes")


    ax[1, 0].imshow(rgb_image) 
    ax[1, 0].set_title("Astrocytes")

    ax[1, 1].set_title("Distance to highest correlation\n(Max 15, astrocytes moving)")
    im = ax[1,1].imshow(dist1, cmap = 'plasma')
    fig.colorbar(im)
    ax[1, 2].set_title("Highest Correlation")
    im = ax[1,2].imshow(m1, cmap = 'plasma')
    fig.colorbar(im)
    ax[1,3].set_title("Direction to\nhighest correlation")
    ax[1,3].imshow(a1)
    ax[1, 4].set_title("Distance to highest correlation\n(Max 15, neurons moving)")
    im = ax[1,4].imshow(dist2, cmap = 'plasma')
    fig.colorbar(im)
    ax[1, 5].set_title("Highest cor")
    im = ax[1,5].imshow(m2, cmap = 'plasma')
    fig.colorbar(im)
    ax[1,6].set_title("Direction to\nhighest correlation")
    ax[1,6].imshow(a2)

    ax[0, 6].set_axis_off()


    tip_xy = tip_location_roi(info["tip_location"], info['grid_size'])
    for im in ax.flatten():
        im.set_xticks([])
        im.set_yticks([])

        im.add_patch(ptch.Rectangle(tip_xy, 1, 1, lw = 1, fc = 'none', ec = 'g'))
    mouse_number = info['mouse_number']
    exp_number = info['exp_number']
    current = info['current']
    fig.savefig(f'/home/s233039/Desktop/KU/results/color_encoding_correlations/{mouse_number}/{exp_number}_{current}_uA.png')





@create_and_save_figure(rows = 3, columns = 5, folder = "peaks_12_4")
def peak_plots(info, neurons_rois, astrocytes_rois, fig, ax):
    ss1 = info["stimulations_starts"][0]
    my_average = lambda signal:np.average(signal[signal > 0])
    my_std = lambda signal: np.std(signal[signal > 0])

    for row in range(2):
        cells = 'neurons' if row == 0 else 'astrocytes'
        rgb_image = get_image(info[cells], cells)
        ax[row,0].imshow(rgb_image)

        rois = neurons_rois if row == 0 else astrocytes_rois
        norm_filtered_rois, normed_rois = normalize_baseline_std(rois, ss1)
        # norm_rois = norm_rois - np.average(norm_rois[:ss1, :, :], axis = 0)
        # norm_rois = np.apply_along_axis(lambda signal: moving_average(signal, 10), axis = 0, arr = norm_rois)

        window = 3
        responsive_peaks = find_activity(norm_filtered_rois, info['stimulations_starts'], info["fs"],window, time_above_threshold=1)
        peaks = find_peaks(normed_rois, info["stimulations_starts"], info["fs"], window)

        # imshow(fig, ax[row, 1], np.average(peaks, axis = 0), vmin = 0, vmax = 30 if row == 0 else None, title = "Average over all peaks")

        # imshow(fig, ax[row, 2], np.std(peaks, axis = 0)/np.average(peaks, axis = 0), vmin = 0, vmax = 1.2, title = "Relative STD\nover all peaks")

        # multiplier = 1
        # get_threshold  = lambda signal: np.average(signal[:ss1]) + multiplier*np.std(signal[:ss1])
        # threshold = np.apply_along_axis(get_threshold, axis = 0, arr = norm_rois)

        imshow(fig, ax[row, 1], np.sum(responsive_peaks, axis = 0), vmin =0, vmax = 10, title = "Number of responsive peaks\n > 2$\sigma_{base}$")
        im2 = overlap_image_map(info[cells], cells, np.sum(responsive_peaks, axis = 0))
        imshow(fig, ax[row, 2], im2, title = "Overlapping\nactive rois", colorbar=False)

        if row == 0:
            thresholded_peaks = peaks * responsive_peaks
            avg = np.apply_along_axis(my_average, axis = 0, arr = thresholded_peaks)
            imshow(fig, ax[row, 3], avg, title = "Average over\nactive peaks")

            std = np.apply_along_axis(my_std, axis = 0, arr = thresholded_peaks)
            imshow(fig, ax[row, 4], std/avg, vmin = 0, vmax = 1.2, title = "Relative STD\nover active peaks")

        tip_xy = tip_location_roi(info['tip_location'], info['grid_size'])
        for i in range(1, 4):
            add_tip_location(ax[row, i], tip_xy)


    for window_shift in (2, 4):
        shifted_responsive_peaks = find_activity(norm_filtered_rois, info["stimulations_starts"], info["fs"], window, window_shift, time_above_threshold=1)
        shifted_peaks = find_peaks(normed_rois, info["stimulations_starts"], info["fs"], window, window_shift)
        imshow(fig, ax[2, window_shift-2], np.sum(shifted_responsive_peaks, axis = 0), vmin = 0, vmax = 10, title = f"Active peaks\nfrom t = {window_shift} to t = {window_shift + window}")
        im = np.apply_along_axis(my_average, axis = 0, arr = shifted_responsive_peaks*shifted_peaks)
        imshow(fig, ax[2, window_shift-1], im, vmin = 0,  title = f"Average of active peaks\nfrom t = {window_shift} to t = {window_shift + window}")
    for i in range(1, 4):
            add_tip_location(ax[2, i], tip_xy)
    im = overlap_image_map(info[cells], cells, np.sum(shifted_responsive_peaks, axis = 0))
    imshow(fig, ax[1,3], im, title = "Overlap with active rois\n from t = 4", colorbar = False)
    im = np.apply_along_axis(my_std, axis = 0, arr = shifted_responsive_peaks*shifted_peaks)
    imshow(fig, ax[2, 4], im, vmax = 2, title = "Std ove active rois")

    ax[1, 4].set_axis_off()

@create_and_save_figure(rows = 1, columns = 5, folder = "astrocytes_rise")
def astrocyte_rise(info, neurons_rois, astrocytes_rois, fig, ax):

    ss1 = info["stimulations_starts"][0]
    def weight_center(signal1:np.ndarray):
        u, s = np.average(signal1[:ss1]), np.std(signal1[:ss1])
        signal1 = (signal1 - u)/s
        window = ss1 if ss1<20 else 20
        mva = np.convolve(signal1, np.full(window, 1/window))
        filt = np.convolve(mva>1, np.full(15, 1/15)) == 1
        m = mva[:astrocytes_rois.shape[0]] * filt[:astrocytes_rois.shape[0]]
        first = np.where(m > 0)
        try:
            first = first[0][0]
        except IndexError:
            first = np.nan
        avg = np.average(m[m>0])
        std = np.std(m[m>0])
        rng = np.sum(m>0)
        if rng == 0:
            rng = np.nan


        return first/info["fs"], rng/info["fs"], avg, std
    maps = np.apply_along_axis(weight_center, axis = 0, arr = astrocytes_rois)
    imshow(fig, ax[0], get_image(info["astrocytes"], "astrocytes"),title = "Astrocytes", colorbar = False)
    imshow(fig, ax[1], maps[0], title = "Start of the increase (s)")
    imshow(fig, ax[2], maps[1], title = "Duration of the increase (s)")
    imshow(fig, ax[3], maps[2], title = "Average")
    imshow(fig, ax[4], maps[3], title = "Std")

@create_and_save_figure(rows = 3, columns = 6, folder = 'segmentation_overlap_1_5')
def overlaping_plots(info, neurons_rois, astrocytes_rois, fig, ax):
    ss1 = info["stimulations_starts"][0]
    norm_filtered_rois_neuorons, normed_rois_neurons = normalize_baseline_std(neurons_rois, ss1)
    norm_filtered_rois_astrocytes, normed_rois_astrocytes = normalize_baseline_std(astrocytes_rois, ss1)

    window = 3
    responsive_peaks = find_activity(norm_filtered_rois_neuorons, info['stimulations_starts'], info["fs"],window, time_above_threshold=1)
    responsive_peaks_neurons = np.sum(responsive_peaks, axis = 0)
    peaks_neurons = find_peaks(normed_rois_neurons, info["stimulations_starts"], info["fs"], window)
    peaks_neurons[responsive_peaks == 0] = np.nan

    responsive_peaks = find_activity(norm_filtered_rois_astrocytes, info['stimulations_starts'], info["fs"],window, time_above_threshold=1)
    responsive_peaks_astrocytes = np.sum(responsive_peaks, axis = 0)
    peaks_astrocytes = find_peaks(normed_rois_astrocytes, info["stimulations_starts"], info["fs"], window)
    peaks_astrocytes[responsive_peaks == 0] = np.nan

    scatter_size = 16
    image = np.average(info['neurons'], axis = 0)
    image = enhance_contrast(image)
    ss1 = info['stimulations_starts'][0]

    imshow(fig, ax[0, 0], get_image(image, 'neurons', True), title = "Neurons", colorbar = False)
    imshow(fig, ax[0, 1], get_image(image, 'neurons', True), title = "Segmented Neurons", colorbar = False)
    segmentation = segment_neurons(image)
    imshow(fig, ax[0, 1], segmentation, cmap = 'gray', alpha = 0.3, colorbar = False)


    seg_rois = rois_from_segmentation(segmentation)
    imshow(fig, ax[0, 2], responsive_peaks_neurons, title = "Number of responsive peaks", vmin = 0, vmax = 10, colorbar=True)
    imshow(fig, ax[0, 2], seg_rois, cmap = 'gray', alpha = 0.3, colorbar = False)
    y = responsive_peaks_neurons.flatten()
    ax[0, 3].scatter(seg_rois.flatten(), y, c = y,  sizes = np.full_like(seg_rois.flatten(), scatter_size), cmap = 'plasma')
    ax[0, 3].set_xlabel("Overlap with segmented neurons [%]")
    ax[0, 3].set_ylabel("Number of responsive peaks")
    
    imshow(fig, ax[0, 4], np.nanmean(peaks_neurons, axis = 0), title= "Average over responsive peaks", colorbar = True)
    y = np.nanmean(peaks_neurons, axis = 0).flatten()
    ax[0, 5].scatter(seg_rois.flatten(),y ,c=y, sizes = np.full_like(seg_rois.flatten(), scatter_size), cmap = 'plasma')
    ax[0, 5].set_xlabel("Overlap with segmented neurons [%]")
    ax[0, 5].set_ylabel("Average over responsive peaks")

    image = np.average(info['astrocytes'], axis = 0)
    image = enhance_contrast(image)

    imshow(fig, ax[1, 0], get_image(image, 'astrocytes', True), title = "Astrocytes", colorbar = False)
    imshow(fig, ax[1, 1], get_image(image, 'astrocytes', True), title = "Segmented Astrocytes", colorbar = False)
    segmentation = segment_astrocytes(image)
    imshow(fig, ax[1, 1], segmentation, cmap = 'gray', alpha = 0.3, colorbar = False)

    seg_rois = rois_from_segmentation(segmentation)
    imshow(fig, ax[1, 2], responsive_peaks_astrocytes, title = "Number of responsive peaks", vmin = 1, vmax = 10, colorbar = True)
    imshow(fig, ax[1, 2], seg_rois, cmap = 'gray', alpha = 0.3, colorbar = False)
    y = responsive_peaks_astrocytes.flatten()
    ax[1, 3].scatter(seg_rois.flatten(), y,c = y,  sizes = np.full_like(seg_rois.flatten(), scatter_size), cmap = 'plasma')
    ax[1, 3].set_xlabel("Overlap with segmented astrocytes [%]")
    ax[1, 3].set_ylabel("Number of responsive peaks")

    
    imshow(fig, ax[1, 4], np.nanmean(peaks_astrocytes, axis = 0))
    y = np.nanmean(peaks_astrocytes, axis = 0).flatten()
    ax[1, 5].scatter(seg_rois.flatten(),y ,c=y, sizes = np.full_like(seg_rois.flatten(), scatter_size), cmap = 'plasma')
    ax[1, 5].set_xlabel("Overlap with segmented astrocytes [%]")
    ax[1, 5].set_ylabel("Average over responsive peaks")


    responsive_peaks = find_activity(norm_filtered_rois_astrocytes, info['stimulations_starts'], info["fs"],window,window_shift=2, time_above_threshold=1)
    peaks_astrocytes = find_peaks(normed_rois_astrocytes, info["stimulations_starts"], info["fs"], window)
    peaks_astrocytes[responsive_peaks == 0] = np.nan
    responsive_peaks_astrocytes = np.sum(responsive_peaks, axis = 0)
    imshow(fig, ax[2, 0], responsive_peaks_astrocytes, cmap = 'plasma', title = "Number of responsive peaks (delay 2s)")
    imshow(fig, ax[2, 0], seg_rois, cmap = 'gray', alpha = 0.3, colorbar = False)
    y = responsive_peaks_astrocytes.flatten()
    ax[2, 1].scatter(seg_rois.flatten(), y,c = y,  sizes = np.full_like(seg_rois.flatten(), scatter_size), cmap = 'plasma')
    ax[2, 1].set_xlabel("Overlap with segmented astrocytes [%]")
    ax[2, 1].set_ylabel("Number of responsive peaks")

    imshow(fig, ax[2, 2], np.nanmean(peaks_astrocytes, axis = 0), cmap = 'plasma', title = "Average over responsive peaks (delay 2s)")
    y = np.nanmean(peaks_astrocytes, axis = 0).flatten()
    ax[2, 3].scatter(seg_rois.flatten(),y ,c=y, sizes = np.full_like(seg_rois.flatten(), 16), cmap = 'plasma')
    ax[2, 3].set_xlabel("Overlap with segmented astrocytes [%]")
    ax[2, 3].set_ylabel("Average over responsive peaks")

    for im in [ ax[2,4], ax[2,5]]:
        im.set_xticks([])
        im.set_yticks([])
        im.axis("off")