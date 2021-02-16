#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Hendrik Heiser
# created on 22 October 2019

import tkinter as tk
from tkinter import filedialog
import os
from glob import glob
import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
from standard_pipeline import preprocess as pre
import place_cell_class as pc
from standard_pipeline.behavior_import import progress
from skimage import io
import tifffile as tif
from datetime import datetime
import shutil
from spike_prediction.spike_prediction import predict_spikes

#%% File and directory handling


def set_file_paths():
    root = tk.Tk()
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)
    root_dir = filedialog.askdirectory(title='Select folder that contains all trial folders')
    root_dir = root_dir.replace('/', '\\')
    root.withdraw()
    print(f'Root directory:\n {root_dir}')
    return root_dir


def save_cnmf(cnm, root=None, path=None, overwrite=False, verbose=True):
    if root is None and path is not None:
        save_path = path
    elif root is not None and path is None:
        save_path = os.path.join(root, 'cnm_results.hdf5')
    else:
        print('Give either a root directory OR a complete path! CNM save directoy ambiguous.')
        return
    if os.path.isfile(save_path) and not overwrite:
        answer = None
        while answer not in ("y", "n", 'yes', 'no'):
            answer = input(f"File [...]{save_path[-40:]} already exists!\nOverwrite? [y/n] ")
            if answer == "yes" or answer == 'y':
                print('Saving...')
                cnm.save(save_path)
                print(f'CNM results successfully saved at {save_path}')
                return save_path
            elif answer == "no" or answer == 'n':
                print('Saving cancelled.')
                return None
            else:
                print("Please enter yes or no.")
    else:
        if verbose:
            print('Saving...')
            cnm.save(save_path)
            print(f'CNM results successfully saved at {save_path}')
        else:
            cnm.save(save_path)


def load_cnmf(root, cnm_filename='cnm_results_manual.hdf5'):
    cnm_filepath = os.path.join(root, cnm_filename)
    cnm_file = glob(cnm_filepath)
    if len(cnm_file) < 1:
        if cnm_filename != 'cnm_results.hdf5':
            return load_cnmf(root, 'cnm_results.hdf5')
        else:
            raise FileNotFoundError(f'No file with the name {cnm_filepath} found.')
    else:
        print(f'Loading file {cnm_file[0]}...')
        return cnmf.load_CNMF(cnm_file[0])


def load_mmap(root, fname=None):
    """
    Loads the motion corrected mmap file of the whole session for CaImAn procedure.
    :param root: folder that holds the session mmap file (should hold only one mmap file)
    :param fname: str, name of the specific mmap file. Necessary if there are multiple mmap files in the same directory.
    :return images: memory-mapped array of the whole session (format [n_frame x X x Y])
    :return str: path to the loaded mmap file
    """
    mmap_file = glob(root + r'\\*.mmap')
    if len(mmap_file) > 1:
        if fname is None:
            raise FileNotFoundError(f'Found more than one mmap file in {root}. Movie could not be loaded.')
        elif os.path.join(root, fname) in mmap_file:
            print(f'Loading file {os.path.join(root, fname)}...')
            Yr, dims, T = cm.load_memmap(os.path.join(root, fname))
            images = np.reshape(Yr.T, [T] + list(dims), order='F')
            return os.path.join(root, fname), images
    elif len(mmap_file) < 1:
        raise FileNotFoundError(f'No mmap file found in {root}.')
    else:
        print(f'Loading file {mmap_file[0]}...')
        Yr, dims, T = cm.load_memmap(mmap_file[0])
        images = np.reshape(Yr.T, [T] + list(dims), order='F')
        return mmap_file[0], images


def load_pcf(root, fname=None):
    if fname is not None:
        pcf_path = glob(os.path.join(root, fname+'.pickle'))
        if len(pcf_path) < 1:
            pcf_path = glob(os.path.join(root, fname))
            if len(pcf_path) < 1:
                raise FileNotFoundError(f'No pcf file found in {os.path.join(root, fname)}.')
    else:
        pcf_path = glob(root + r'\\pcf_results*')
        if len(pcf_path) < 1:
            raise FileNotFoundError(f'No pcf file found in {root}.')
        elif len(pcf_path) > 1:
            pcf_path = glob(root + r'\\pcf_results_manual.pickle')
            if len(pcf_path) < 1:
                pcf_path = glob(root + r'\\pcf_results.pickle')
                if len(pcf_path) < 1:
                    pcf_path = glob(root + r'\\pcf_results')
                elif len(pcf_path) > 1:
                    raise FileNotFoundError(f'More than one pcf file found in {root}.')
    print(f'Loading file {pcf_path[0]}...')
    with open(pcf_path[0], 'rb') as file:
        obj = pickle.load(file)

    return obj


def load_manual_neuron_coordinates(path, fname=None):
    """
    Loads the .txt table containing coordinates of manually selected neurons through Adrians GUI. Files have the file
    name 'clicked_neurons_[DATE]_[TIME].txt' to prevent overwriting of multiple annotation sessions. If a specific
    fname is not provided and multiple files are found, the most recent file is loaded by default.
    :param path: str, path to the session directory
    :param fname: str, optional, specific filename of the file to be loaded. If None, the most recent file is loaded.
    :return: list of tuples of (x, y) coordinates of manually selected neurons
    """

    file_list = glob(path+r'\clicked_neurons_*.txt')
    if fname is not None:
        if os.path.join(path, fname) in file_list:
            filename = os.path.join(path, fname)
            print(f'Loading specific .txt file: {fname}!')
        else:
            return FileNotFoundError(f'No file found at {os.path.join(path, fname)}.')
    else:
        if len(file_list) == 1:
            filename = file_list[0]
            print(f'Loading .txt file: {filename[-35:]}!')
        elif len(file_list) > 1:
            dates = [datetime.strptime(x[-19:-4], '%Y%m%d_%H%M%S') for x in file_list]  # Get date and time of files
            filename = file_list[dates.index(max(dates))]  # Get path of most recent file
            print(f'Loading most recent .txt file: {filename[-35:]}!')
        elif len(file_list) == 0:
            return FileNotFoundError(f'No files found at {path}.')
        else:
            return ValueError(f'Loading of clicked_neuron.txt at {path} unsuccessful...')

    coords = np.loadtxt(filename, delimiter='\t', skiprows=1, dtype='int16')
    return list(zip(coords[:, 1], coords[:, 2]))


def export_tif(root, target_folder=None):
    """ Export a motion corrected memory mapped file to an ImageJ readable .tif stack
    :param root: str
        Path of the folder containing the mmap file
    :param target_folder: str (default None)
        Destination folder of the exported .tif. If None, use folder of the .mmap file
    Adrian 2019-03-21
    """

    mmap_file, movie = load_mmap(root)
    file_name = os.path.splitext(os.path.basename(mmap_file))[0] + '.tif'
    print(f'Start saving memmap movie to TIFF...')

    if target_folder is not None:
        root = target_folder

    # Transform movie to 16 bit int
    movie_int = np.array(movie, dtype='int16' )
    movie = None   # save memory

    # Transform into C-order
    toSave_cOrder = movie_int.copy(order='C')
    movie_int = None   # save memory

    # Save movie to the specified path
    tif.imsave(os.path.join(root, file_name), data=toSave_cOrder)
    print('Done!')

#%% CNMF wrapper functions

# todo: make work for very large movies (> 45,000 frames), maybe by loading movie in F-order!
def get_local_correlation(movie):
    """
    Calculates local correlation map of a movie.
    :param movie:  mmap file of the movie with the dimensions [n_frames x X x Y]
    :return lcm: local correlation map
    """
    lcm = cm.local_correlations(movie, swap_dim=False)
    lcm[np.isnan(lcm)] = 0
    return lcm

def get_local_correlation_sequential(mc):
    """
    Calculates local correlation map of a movie with more than 40000 frames. For normal lcm construction, the whole
    movie has to be loaded into memory. To avoid this for large files, single-trial files are loaded sequentially and
    the lcm is the mean of all trials.
    :param mc:  motioncorrect object that holds paths to single-trial mmap files
    :return lcm: local correlation map
    """
    lcm_array = np.zeros((len(mc.mmap_file), mc.total_template_els.shape[0], mc.total_template_els.shape[1]))
    for trial, file in enumerate(mc.mmap_file):
        Yr, dims, T = cm.load_memmap(file)
        images = np.reshape(Yr.T, [T] + list(dims), order='F')
        lcm_array[trial] = get_local_correlation(images)
    return np.mean(lcm_array, axis=0)

def save_local_correlation(movie, path, sequential=False):
    """
    Calls get_local_correlation() for small files or get_local_correlation_sequential() for files > 40'000 frames and
    saves it as a TIFF in the provided directory.
    :param movie: mmap file of the session-movie (if sequential=False) or MC object (if sequential=True)
    :param path: session directory where the LCM should be saved
    :param sequential: bool flag whether to get LCM from whole-session file or from single-trial files
    :return:
    """
    if sequential:
        cor = get_local_correlation_sequential(movie)
    else:
        cor = get_local_correlation(movie)
    fname = path + r'\local_correlation_image.tif'
    io.imsave(fname, cor.astype('float32'))
    print(f'Saved local correlation image at {fname}.')
    return fname


def save_average_image(movie, path, sequential=False):
    if sequential:
        avg_array = np.zeros((len(movie.mmap_file), movie.total_template_els.shape[0], movie.total_template_els.shape[1]))
        for trial, file in enumerate(movie.mmap_file):
            Yr, dims, T = cm.load_memmap(file)
            images = np.reshape(Yr.T, [T] + list(dims), order='F')
            for row in range(images.shape[1]):
                for col in range(images.shape[2]):
                    curr_pix = images[:, row, col]
                    avg_array[trial, row, col] = np.mean(curr_pix)
        avg = np.mean(avg_array, axis=0)
    else:
        avg = np.zeros(movie.shape[1:])
        for row in range(movie.shape[1]):
            for col in range(movie.shape[2]):
                curr_pix = movie[:, row, col]
                avg[row, col] = np.mean(curr_pix)
    fname = path + r'\mean_intensity_image.tif'
    io.imsave(fname, avg.astype('float32'))
    print(f'Saved mean intensity image at {fname}.')
    return fname


def run_evaluation(images, cnm, dview):
    cnm.estimates.evaluate_components(images, params=cnm.params, dview=dview)
    return cnm


def run_source_extraction(images, params, dview):
    """
    Wrapper function for CaImAn source extraction. Takes the mmap movie file and the CNMFParams object to perform
    source extraction and deconvolution.
    :param images: mmap file of the movie with the dimensions [n_frames x X x Y]
    :param params: CNMFParams object holding all required parameters
    :return cnm object with extracted components
    """
    temp_p = params.get('temporal', 'p')
    params = params.change_params({'p': 0})
    cnm = cnmf.CNMF(params.get('patch', 'n_processes'), params=params, dview=dview)
    cnm = cnm.fit(images)
    cnm.params.change_params({'p': temp_p})
    cnm2 = cnm.refit(images, dview=dview)
    cnm2.estimates.dims = images.shape[1:]
    return cnm2


# def run_neuron_selection_gui(path):
#     """
#     Runs Adrians GUI for manually selecting neurons from a session's .mmap file.
#     :param path: str, directory of an imaging session that holds the corresponding .mmap file
#     :return:
#     """
#     import subprocess
#     path = r'W:\\Neurophysiology-Storage1\\Wahl\\Hendrik\\PhD\\Data\\Batch3\\M37\\20200318'
#     script_path = r'C:\\Users\\hheise\\PycharmProjects\\Caiman\\custom scripts\\manual_neuron_selection_gui.py'
#     command = ['python', script_path, '--session', path]
#     stout = subprocess.run(command)


def manual_neuron_extraction(root, movie, params, dview, fname=None):
    """
    Run Caimans source extraction with neurons manually selected by Adrians selection GUI.
    :param root: str, path of session directory
    :param movie: mmap file of this session's movie
    :param params: CNMFParams object
    :param dview: link to Caiman processing cluster
    :param fname: str, optional, specific clicked_neurons.txt file to load (if several for one session exist).
    :return:
    """

    # Load coordinates of manually selected neurons (saved by Adrians GUI in a .txt table)
    coords = load_manual_neuron_coordinates(root, fname)

    dims = movie.shape[1:]                                      # Get dimensions of movie
    A = np.zeros((np.prod(dims), len(coords)), dtype=bool)      # Prepare array that will hold all neuron masks
    neuron_half_size = params.init['gSig'][0]                   # Get expected half-size in pixels of neurons

    # Set kernel of neuronal shape (circle with a diameter the size of expected neurons)
    import skimage.morphology
    kernel = skimage.morphology.disk(radius=neuron_half_size - 0.1)  # -0.1 to remove single pixels

    # Create a mask for each neuron (circle around the coordinates) and add the flattened version to A
    for i, neuron in enumerate(coords):
        mask = np.zeros(shape=dims)
        x = neuron[0]
        y = neuron[1]
        # create circle around neuron location
        mask[y, x] = 1
        mask = skimage.morphology.dilation(mask, kernel)
        mask = mask == 1  # change to boolean type

        # flatten the mask to a 1D array and add it to A
        A[:, i] = mask.flatten('F')

    # make sure the caiman parameter are set correctly for manual masks
    params.set('patch', {'only_init': False})
    params.set('patch', {'rf': None})

    # # with fit and refit
    # temp_p = params.get('temporal', 'p')
    # params = params.change_params({'p': 0})
    # cnm = cnmf.CNMF(params.get('patch', 'n_processes'), params=params, dview=dview)
    # cnm = cnm.fit(movie)
    # cnm.params.change_params({'p': temp_p})
    # cnm2 = cnm.refit(movie, dview=dview)
    # cnm2.estimates.dims = movie.shape[1:]
    # return cnm2

    # run refinement of masks and extraction of traces
    # params = params.change_params({'p': 0})
    cnm = cnmf.CNMF(params.get('patch', 'n_processes'), params=params, dview=dview, Ain=A)  # A is passed to CNMF object
    cnm.fit(movie)
    cnm.estimates.dims = movie.shape[1:]

    return cnm

"""   Not really working, FOV shift is too unreliable in some parts of the FOV to automatically select neurons
def import_template_coordinates(curr_path, temp_path):

    # Load template coordinates from .txt file
    template_coords = load_manual_neuron_coordinates(temp_path)

    # Load mean intensity and local correlation images from current and template sessions
    avg_curr = io.imread(curr_path + r'\mean_intensity_image.tif')
    avg_temp = io.imread(temp_path + r'\mean_intensity_image.tif')
    cor_curr = io.imread(curr_path + r'\local_correlation_image.tif')
    cor_temp = io.imread(temp_path + r'\local_correlation_image.tif')


    # Compute piecewise shift of current vs template images
    from multisession_registration import piecewise_fov_shift, shift_com
    x_shift_avg, y_shift_avg = piecewise_fov_shift(avg_temp, avg_curr, n_patch=1)
    x_shift_cor, y_shift_cor = piecewise_fov_shift(cor_temp, cor_curr)

    # Shift coordinates according to the map
    temp_coord_shift_avg = [shift_com(x, (x_shift_avg[x[0], x[1]], y_shift_avg[x[0], x[1]]), avg_temp.shape) for x in template_coords]


    fig, ax = plt.subplots(1, 2, True, True)
    ax[0].imshow(avg_curr)
    ax[1].imshow(avg_temp)
    for x_temp, x_shift in zip(template_coords, temp_coord_shift_avg):
        cross = ax[1].plot(x_temp[0], x_temp[1], 'x', color='red')         # Plot cross on template average image (should fit)
        cross = ax[0].plot(x_temp[0], x_temp[1], 'x', color='red')         # Plot cross on current average image (should not fit)
        cross = ax[0].plot(x_shift[0], x_shift[1], 'x', color='yellow')    # Plot adjusted cross on current average image (should fit)
"""
#%% Motion correction wrapper functions



def motion_correction(root, params, dview, percentile=0.01, temp_dir=r'C:\Users\hheise\temp_files',
                      remove_f_order=True, remove_c_order=True, get_images=True, overwrite=False):
    """
    Wrapper function that performs motion correction, saves it as C-order files and can immediately remove F-order files
    to save disk space. Function automatically finds sessions and performs correction on whole sessions separately.
    :param root: str; path in which imaging sessions are searched (files should be in separate trial folders)
    :param params: cnm.params object that holds all parameters necessary for motion correction
    :param dview: link to Caimans processing server
    :param temp_dir: str, folder on computer hard disk where temporary files are saved
    :param percentile: float, percentile that should be added to the .tif files to avoid negative pixel values
    :param remove_f_order: bool flag whether F-order files should be removed to save disk space
    :param remove_c_order: bool flag whether single-trial C-order files should be removed to save disk space
    :param get_images: bool flag whether local correlation and mean intensity images should be computed after mot corr
    :param overwrite: bool flag whether correction should be performed even if a memmap file already exists
    :return mmap_list: list that includes paths of mmap files for all processed sessions
    """

    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split('(\d+)', text)]

    # First, get a list of all folders that include contiguous imaging sessions (have to be motion corrected together)
    dir_list = []
    for step in os.walk(root):
        if len(glob(step[0] + r'\\file_00???.tif')) > 0:
            up_dir = step[0].rsplit(os.sep, 1)[0]
            if ((len(glob(up_dir + r'\\memmap__d1_*.mmap')) == 0 and len(glob(up_dir + r'\\pcf*')) == 0 and
                len(glob(up_dir + r'\\cnm*')) == 0) or overwrite) and up_dir not in dir_list and 'bad_trials' not in up_dir:
                dir_list.append(up_dir)   # this makes a list of all folders that contain single-trial imaging folders

    mmap_list = []
    if len(dir_list) > 0:

        if remove_f_order:
            print('\nF-order files will be removed after processing.')
        if remove_c_order:
            print('Single-trial C-order files will be removed after processing.')
        if get_images:
            print('Local correlation and mean intensity images will be created after processing.')

        print(f'\nFound {len(dir_list)} sessions that have not yet been motion corrected:')
        for session in dir_list:
            print(f'{session}')

        # Then, perform motion correction for each of the session
        for session in dir_list:

            # restart cluster
            cm.stop_server(dview=dview)
            c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)

            # list of all .tif files of that session which should be corrected together, sorted by their trial number
            file_list = glob(session + r'\\*\\*_00???.tif')
            file_list.sort(key=natural_keys)
            file_list = [x for x in file_list if 'wave' not in x]   # ignore files with waves (disrupt ROI detection)
            print(f'\nNow starting to process session {session} ({len(file_list)} trials).')

            # Preprocessing
            temp_files = []
            for raw_file in file_list:
                ### Preprocessing steps from Adrian:
                ### (https://github.com/HelmchenLabSoftware/adrian_pipeline/blob/master/schema/img.py#L401)
                stack = io.imread(raw_file)

                # correct stack and also crop artifact on left side of image
                stack = pre.correct_line_shift_stack(stack, crop_left=20, crop_right=20)

                # Make movie positive (negative values crash NON-NEGATIVE matrix factorisation)
                stack = stack - int(np.percentile(stack, percentile))

                fname = os.path.splitext(os.path.basename(raw_file))[0]
                if not os.path.isfile(os.path.join(temp_dir, fname+'_corrected.tif')):  # avoid overwriting
                    new_path = os.path.join(temp_dir, fname+'_corrected.tif')
                else:
                    new_path = os.path.join(temp_dir, fname + '_corrected_2.tif')
                tif.imwrite(new_path, data=stack)
                temp_files.append(new_path)

            # Perform motion correction
            mc = MotionCorrect(temp_files, dview=dview, **params.get_group('motion'))
            mc.motion_correct(save_movie=True)
            border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0

            # memory map the file in order 'C'
            print(f'Finished motion correction. Starting to save files in C-order...')
            fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C', border_to_0=border_to_0)
            mmap_list.append(fname_new)

            # compute local correlation and mean intensity image if they do not already exist
            if get_images:
                print(f'Finished. Now computing local correlation and mean intensity images...')
                if eval(os.path.basename(fname_new).split(sep='_')[-2]) > 40000:
                    if not os.path.isfile(os.path.join(session, 'local_correlation_image.tif')):
                        out = save_local_correlation(mc, session, sequential=True)
                    if not os.path.isfile(os.path.join(session, 'mean_intensity_image.tif')):
                        out = save_average_image(mc, session, sequential=True)
                else:
                    Yr, dims, T = cm.load_memmap(fname_new)
                    images = np.reshape(Yr.T, [T] + list(dims), order='F')
                    if not os.path.isfile(os.path.join(session, 'local_correlation_image.tif')):
                        out = save_local_correlation(images, session)
                    if not os.path.isfile(os.path.join(session, 'mean_intensity_image.tif')):
                        out = save_average_image(images, session)

                    # close opened mmap file to enable moving file
                    del Yr, images

            # transfer final file to target directory on the server
            target_path = os.path.join(session, os.path.basename(fname_new))
            shutil.move(fname_new, target_path)

            # clear up temporary files (corrected TIFFs, F and C-order mmap files)
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))

            print('Finished!')

    else:
        print('Found no sessions to motion correct!')

    return mmap_list, dview



#%% Spatial information

def si_formula(data, position, n_bins=60):
    """
    True function that actually calculates spatial information. Data and position should be preprocessed.
    :param data: np.array containing neural data (spike probabilities); shape (#samples x #neurons)
    :param position: np.array containing position data for every sample; shape (#samples x 1)
    :param n_bins: number of bins the data should be binned into. Default is 60.
    :return spatial_info: 1D np.array with shape (#neurons) containing SI value for every neuron
    """
    # bin data into n_bins, get mean event rate per bin
    bin_borders = np.linspace(int(min(position)), int(max(position)), n_bins)
    idx = np.digitize(position, bin_borders)  # get indices of bins

    # get fraction of bin occupancy
    unique_elements, counts_elements = np.unique(idx, return_counts=True)
    bin_freq = np.array([x / np.sum(counts_elements) for x in counts_elements])

    # get mean spikes/s for each bin
    bin_mean_act = np.zeros((n_bins, data.shape[1]))
    for bin_nr in range(n_bins):
        curr_bin_idx = np.where(idx == bin_nr + 1)[0]
        bin_act = data[curr_bin_idx]
        bin_mean_act[bin_nr, :] = np.sum(bin_act, axis=0) / (bin_act.shape[0] / 30)  # firing rate per second
        # bin_mean_act[bin_nr, :] = np.mean(bin_act, axis=0)    # firing rate per frame
    total_firing_rate = np.sum(data, axis=0) / (data.shape[0] / 30)
    # total_firing_rate = np.mean(data, axis=0)

    # calculate spatial information content
    spatial_info = np.zeros(len(total_firing_rate))
    for cell in range(len(total_firing_rate)):
        curr_trace = bin_mean_act[:, cell]
        tot_act = total_firing_rate[cell]  # this is the total dF/F averaged across all bins
        bin_si = np.zeros(n_bins)  # initialize array that holds SI value for each bin
        for i in range(n_bins):
            # apply the SI formula to every bin
            if curr_trace[i] <= 0 or tot_act <= 0:
                bin_si[i] = np.nan
            else:
                bin_si[i] = curr_trace[i] * np.log2(curr_trace[i] / tot_act) * bin_freq[i]
        if np.all(np.isnan(bin_si)):
            spatial_info[cell] = np.nan
        else:
            spatial_info[cell] = np.nansum(bin_si)

    return spatial_info


def get_spatial_info(all_data, behavior, n_bootstrap=2000):

    # %% remove samples where the mouse was stationary (less than 30 movement per frame)
    all_position = []
    behavior_masks = []
    for trial in behavior:
        all_position.append(trial[np.where(trial[:, 3] == 1), 1])  # get position of mouse during every frame
        behavior_masks.append(np.ones(int(np.nansum(trial[:, 3])), dtype=bool))  # bool list for every frame of that trial
        frame_idx = np.where(trial[:, 3] == 1)[0]  # find sample_idx of all frames
        for i in range(len(frame_idx)):
            if i != 0:
                if np.nansum(trial[frame_idx[i - 1]:frame_idx[i], 4]) > -30:  # make index of the current frame False if
                    behavior_masks[-1][i] = False  # the mouse didn't move much during the frame
            else:
                if trial[0, 4] > -30:
                    behavior_masks[-1][i] = False
    all_position = np.hstack(all_position).T
    # trial_lengths = [int(np.sum(trial)) for trial in behavior_masks]

    behavior_mask = np.hstack(behavior_masks)
    data = all_data[behavior_mask]
    position = all_position[behavior_mask]

    # dimension extension necessary if input data is only from one neuron
    if len(data.shape) == 1:
        data = data[..., np.newaxis]
    # remove data points where decoding was nan (beginning and end)
    nan_mask = np.isnan(data[:, 0])
    data = data[~nan_mask]
    position = position[~nan_mask]

    # get SI of every cell
    si_raw = si_formula(data, position)

    if n_bootstrap > 0:
        # Bootstrapping:
        # Create new data trace by taking a random value of original data and adding it to a random idx of a new array.
        # Do this X times for every neuron. Normalize raw SI by the average bootstrapped SI for every neuron.
        si_norm = np.zeros((len(si_raw), 2))
        for cell in range(len(si_norm)):
            progress(cell + 1, len(si_norm), status=f'Performing bootstrapping (cell {cell + 1}/{len(si_norm)})')
            orig_data = data[:, cell]
            cell_boot = np.zeros((len(orig_data), n_bootstrap))
            for j in range(n_bootstrap):
                # get a new trace from the original one by randomly selecting data points for every index
                new_trace = np.zeros(len(orig_data))  # initialize empty array for new trace
                for idx in range(len(orig_data)):  # go through entries in new_trace
                    rand_idx = np.random.randint(0, len(orig_data) - 1)  # get a random idx to fill in the current entry
                    new_trace[idx] = orig_data[
                        rand_idx]  # fill the current entry with the random entry of the orig data
                # get spatial information of the newly constructed trace and add it to the cell-wide array
                cell_boot[:, j] = new_trace
            # perform bootstrapping (pretend that every random trace is a different cell)
            si_boot = si_formula(cell_boot, position)
            # after bootstrapping, normalize the SI of the current cell by the average bootstrapped SI
            si_norm[cell, 0] = si_raw[cell] / np.mean(si_boot)
            # add percentage of higher bootstrap SI than original SI
            si_norm[cell, 1] = sum(si > si_raw[cell] for si in si_boot) / n_bootstrap
        return si_norm

    else:
        return si_raw


def check_eval_results(cnm, idx, plot_contours=False):
    """Checks results of component evaluation and determines why the component got rejected or accepted

    Args:
        cnm:                caiman CNMF object containing estimates and evaluate_components() results

        idx:                int or iterable (array, list...)
                            index or list of indices of components to be checked

    Returns:
        printout of evaluation results
    """
    try:
        iter(idx)
        idx = list(idx)
    except:
        idx = [idx]
    snr_min = cnm.params.quality['SNR_lowest']
    snr_max = cnm.params.quality['min_SNR']
    r_min = cnm.params.quality['rval_lowest']
    r_max = cnm.params.quality['rval_thr']
    cnn_min = cnm.params.quality['cnn_lowest']
    cnn_max = cnm.params.quality['min_cnn_thr']

    for i in range(len(idx)):
        snr = cnm.estimates.SNR_comp[idx[i]]
        r = cnm.estimates.r_values[idx[i]]
        cnn = cnm.estimates.cnn_preds[idx[i]]
        cnn_round = str(round(cnn, 2))

        red_start = '\033[1;31;49m'
        red_end = '\033[0;39;49m'

        green_start = '\033[1;32;49m'
        green_end = '\033[0;39;49m'

        upper_thresh_failed = 0
        lower_thresh_failed = False

        print(f'Checking component {idx[i]}...')
        if idx[i] in cnm.estimates.idx_components:
            print(green_start+f'\nComponent {idx[i]} got accepted, all lower threshold were passed!'+green_end+'\n\n\tUpper thresholds:\n')

            if snr >= snr_max:
                print(green_start+f'\tSNR of {round(snr,2)} exceeds threshold of {snr_max}\n'+green_end)
            else:
                print(f'\tSNR of {round(snr,2)} does not exceed threshold of {snr_max}\n')

            if r >= r_max:
                print(green_start+f'\tR-value of {round(r,2)} exceeds threshold of {r_max}\n'+green_end)
            else:
                print(f'\tR-value of {round(r,2)} does not exceed threshold of {r_max}\n')

            if cnn >= cnn_max:
                print(green_start+'\tCNN-value of '+cnn_round+f' exceeds threshold of {cnn_max}\n'+green_end)
            else:
                print('\tCNN-value of '+cnn_round+f' does not exceed threshold of {cnn_max}\n')
            print(f'\n')

        else:
            print(f'\nComponent {idx[i]} did not get accepted. \n\n\tChecking thresholds:\n')

            if snr >= snr_max:
                print(green_start+f'\tSNR of {round(snr,2)} exceeds upper threshold of {snr_max}\n'+green_end)
            elif snr >= snr_min and snr < snr_max:
                print(f'\tSNR of {round(snr,2)} exceeds lower threshold of {snr_min}, but not upper threshold of {snr_max}\n')
                upper_thresh_failed += 1
            else:
                print(red_start+f'\tSNR of {round(snr,2)} does not pass lower threshold of {snr_min}\n'+red_end)
                lower_thresh_failed = True

            if r >= r_max:
                print(green_start+f'\tR-value of {round(r,2)} exceeds upper threshold of {r_max}\n'+green_end)
            elif r >= r_min and r < r_max:
                print(f'\tR-value of {round(r,2)} exceeds lower threshold of {r_min}, but not upper threshold of {r_max}\n')
                upper_thresh_failed += 1
            else:
                print(f'\tR-value of {round(r,2)} does not pass lower threshold of {r_min}\n')
                lower_thresh_failed = True

            if cnn >= cnn_max:
                print(green_start+'\tCNN-value of '+cnn_round+f' exceeds threshold of {cnn_max}\n'+green_end)
            elif cnn >= cnn_min and cnn < cnn_max:
                print('\tCNN-value of '+cnn_round+f' exceeds lower threshold of {cnn_min}, but not upper threshold of {cnn_max}\n')
                upper_thresh_failed += 1
            else:
                print(red_start+'\tCNN-value of '+cnn_round+f' does not pass lower threshold of {cnn_min}\n'+red_end)
                lower_thresh_failed = True

            if lower_thresh_failed:
                print(red_start+f'Result: Component {idx[i]} got rejected because it failed at least one lower threshold!\n\n'+red_end)
            elif upper_thresh_failed == 3 and not lower_thresh_failed:
                print(red_start+f'Result: Component {idx[i]} got rejected because it met all lower, but no upper thresholds!\n\n'+red_end)
            else:
                print('This should not appear, check code logic!\n\n')

    if plot_contours:
        plt.figure()
        out = cm.utils.visualization.plot_contours(cnm.estimates.A[:, idx], cnm.estimates.Cn,
                                                   display_numbers=False, colors='r')


def reject_cells(cnm, idx):
    mask = np.ones(len(cnm.estimates.idx_components), dtype=bool)
    mask[idx] = False

    bad_cells = cnm.estimates.idx_components[~mask]

    cnm.estimates.idx_components_bad = np.concatenate((cnm.estimates.idx_components_bad, bad_cells))
    cnm.estimates.idx_components = cnm.estimates.idx_components[mask]
    return cnm


def accept_cells(cnm, idx):
    mask = np.ones(len(cnm.estimates.idx_components_bad), dtype=bool)
    mask[idx] = False

    good_cells = cnm.estimates.idx_components_bad[~mask]

    cnm.estimates.idx_components = np.concatenate((cnm.estimates.idx_components, good_cells))
    cnm.estimates.idx_components_bad = cnm.estimates.idx_components_bad[mask]
    return cnm


def perform_whole_pipeline(root):

    from caiman.source_extraction import cnmf

    def set_params(mouse):
        if mouse == 'M32':
            # dataset dependent parameters
            fr = 30  # imaging rate in frames per second
            decay_time = 0.4  # length of a typical transient in seconds (0.4)
            dxy = (1.66,
                   1.52)  # spatial resolution in x and y in (um per pixel) [(1.66, 1.52) for 1x, (0.83, 0.76) for 2x]

            # extraction parameters
            p = 1  # order of the autoregressive system
            gnb = 2  # number of global background components (3)
            merge_thr = 0.75  # merging threshold, max correlation allowed (0.86)
            rf = 25  # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
            stride_cnmf = 10  # amount of overlap between the patches in pixels (20)
            K = 15  # number of components per patch (10)
            gSig = [5, 5]  # expected half-size of neurons in pixels [X, Y] (has to be int, not float!)
            method_init = 'greedy_roi'  # initialization method (if analyzing dendritic data using 'sparse_nmf')
            ssub = 2  # spatial subsampling during initialization
            tsub = 2  # temporal subsampling during intialization

            # evaluation parameters
            min_SNR = 9  # signal to noise ratio for accepting a component (default 2)
            SNR_lowest = 3.2
            rval_thr = 0.82  # space correlation threshold for accepting a component (default 0.85)
            rval_lowest = -1
            cnn_thr = 0.99  # threshold for CNN based classifier (default 0.99)
            cnn_lowest = 0.02  # neurons with cnn probability lower than this value are rejected (default 0.1)

            opts_dict = {'fnames': None, 'fr': fr, 'decay_time': decay_time, 'dxy': dxy, 'nb': gnb, 'rf': rf,
                         'K': K,
                         'gSig': gSig, 'stride': stride_cnmf, 'method_init': method_init, 'rolling_sum': True,
                         'merge_thr': merge_thr, 'only_init': True, 'ssub': ssub, 'tsub': tsub,
                         'SNR_lowest': SNR_lowest, 'cnn_lowest': cnn_lowest, 'min_SNR': min_SNR,
                         'min_cnn_thr': cnn_thr,
                         'rval_lowest': rval_lowest, 'rval_thr': rval_thr, 'use_cnn': True}

            cnm_params = cnmf.params.CNMFParams(params_dict=opts_dict)

        elif mouse == 'M33':
            # dataset dependent parameters
            fr = 30  # imaging rate in frames per second
            decay_time = 0.4  # length of a typical transient in seconds (0.4)
            dxy = (1.66,
                   1.52)  # spatial resolution in x and y in (um per pixel) [(1.66, 1.52) for 1x, (0.83, 0.76) for 2x]

            # extraction parameters
            p = 1  # order of the autoregressive system
            gnb = 2  # number of global background components (3)
            merge_thr = 0.75  # merging threshold, max correlation allowed (0.86)
            rf = 25  # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
            stride_cnmf = 10  # amount of overlap between the patches in pixels (20)
            K = 12  # number of components per patch (10)
            gSig = [5, 5]  # expected half-size of neurons in pixels [X, Y] (has to be int, not float!)
            method_init = 'greedy_roi'  # initialization method (if analyzing dendritic data using 'sparse_nmf')
            ssub = 2  # spatial subsampling during initialization
            tsub = 2  # temporal subsampling during intialization

            min_SNR = 6  # signal to noise ratio for accepting a component (default 2)
            SNR_lowest = 2.5
            rval_thr = 0.85  # space correlation threshold for accepting a component (default 0.85)
            rval_lowest = -1
            cnn_thr = 0.95  # threshold for CNN based classifier (default 0.99)
            cnn_lowest = 0.03  # neurons with cnn probability lower than this value are rejected (default 0.1)

            opts_dict = {'fnames': None, 'fr': fr, 'decay_time': decay_time, 'dxy': dxy, 'nb': gnb, 'rf': rf,
                         'K': K,
                         'gSig': gSig, 'stride': stride_cnmf, 'method_init': method_init, 'rolling_sum': True,
                         'merge_thr': merge_thr, 'only_init': True, 'ssub': ssub, 'tsub': tsub,
                         'SNR_lowest': SNR_lowest, 'cnn_lowest': cnn_lowest, 'min_SNR': min_SNR,
                         'min_cnn_thr': cnn_thr,
                         'rval_lowest': rval_lowest, 'rval_thr': rval_thr, 'use_cnn': True}

            cnm_params = cnmf.params.CNMFParams(params_dict=opts_dict)

        elif mouse == 'M37':
            # dataset dependent parameters
            fr = 30  # imaging rate in frames per second
            decay_time = 0.4  # length of a typical transient in seconds (0.4)
            dxy = (1.66,
                   1.52)  # spatial resolution in x and y in (um per pixel) [(1.66, 1.52) for 1x, (0.83, 0.76) for 2x]

            # extraction parameters
            p = 1  # order of the autoregressive system
            gnb = 2  # number of global background components (3)
            merge_thr = 0.75  # merging threshold, max correlation allowed (0.86)
            rf = 25  # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
            stride_cnmf = 10  # amount of overlap between the patches in pixels (20)
            K = 12  # number of components per patch (10)
            gSig = [5, 5]  # expected half-size of neurons in pixels [X, Y] (has to be int, not float!)
            method_init = 'greedy_roi'  # initialization method (if analyzing dendritic data using 'sparse_nmf')
            ssub = 2  # spatial subsampling during initialization
            tsub = 2  # temporal subsampling during intialization

            # evaluation parameters
            min_SNR = 5  # signal to noise ratio for accepting a component (default 2)
            SNR_lowest = 2
            rval_thr = 0.8  # space correlation threshold for accepting a component (default 0.85)
            rval_lowest = 0.4
            cnn_thr = 0.92  # threshold for CNN based classifier (default 0.99)
            cnn_lowest = 0.1  # neurons with cnn probability lower than this value are rejected (default 0.1)

            opts_dict = {'fnames': None, 'fr': fr, 'decay_time': decay_time, 'dxy': dxy, 'nb': gnb, 'rf': rf,
                         'K': K,
                         'gSig': gSig, 'stride': stride_cnmf, 'method_init': method_init, 'rolling_sum': True,
                         'merge_thr': merge_thr, 'only_init': True, 'ssub': ssub, 'tsub': tsub,
                         'SNR_lowest': SNR_lowest, 'cnn_lowest': cnn_lowest, 'min_SNR': min_SNR,
                         'min_cnn_thr': cnn_thr,
                         'rval_lowest': rval_lowest, 'rval_thr': rval_thr, 'use_cnn': True}

            cnm_params = cnmf.params.CNMFParams(params_dict=opts_dict)

        elif mouse == 'M38':
            # dataset dependent parameters
            fr = 30  # imaging rate in frames per second
            decay_time = 0.4  # length of a typical transient in seconds (0.4)
            dxy = (1.66, 1.52)  # spatial resolution (um per pixel) [(1.66, 1.52) for 1x, (0.83, 0.76) for 2x]

            # extraction parameters
            p = 1  # order of the autoregressive system
            gnb = 2  # number of global background components (3)
            merge_thr = 0.75  # merging threshold, max correlation allowed (0.86)
            rf = 25  # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
            stride_cnmf = 10  # amount of overlap between the patches in pixels (20)
            K = 10  # number of components per patch (10)
            gSig = [6, 6]  # expected half-size of neurons in pixels [X, Y] (has to be int, not float!)
            method_init = 'greedy_roi'  # initialization method (if analyzing dendritic data using 'sparse_nmf')
            ssub = 2  # spatial subsampling during initialization
            tsub = 2  # temporal subsampling during intialization

            # evaluation parameters
            min_SNR = 6  # signal to noise ratio for accepting a component (default 2)
            SNR_lowest = 2
            rval_thr = 0.8  # space correlation threshold for accepting a component (default 0.85)
            rval_lowest = -1
            cnn_thr = 0.95  # threshold for CNN based classifier (default 0.99)
            cnn_lowest = 0.15  # neurons with cnn probability lower than this value are rejected (default 0.1)

            opts_dict = {'fnames': None, 'fr': fr, 'decay_time': decay_time, 'dxy': dxy, 'nb': gnb, 'rf': rf,
                         'K': K,
                         'gSig': gSig, 'stride': stride_cnmf, 'method_init': method_init, 'rolling_sum': True,
                         'merge_thr': merge_thr, 'only_init': True, 'ssub': ssub, 'tsub': tsub,
                         'SNR_lowest': SNR_lowest, 'cnn_lowest': cnn_lowest, 'min_SNR': min_SNR,
                         'min_cnn_thr': cnn_thr,
                         'rval_lowest': rval_lowest, 'rval_thr': rval_thr, 'use_cnn': True}

            cnm_params = cnmf.params.CNMFParams(params_dict=opts_dict)

        elif mouse == 'M39':
            # dataset dependent parameters
            fr = 30  # imaging rate in frames per second
            decay_time = 0.4  # length of a typical transient in seconds (0.4)
            dxy = (1.66, 1.52)  # spatial resolution (um per pixel) [(1.66, 1.52) for 1x, (0.83, 0.76) for 2x]

            # extraction parameters
            p = 1  # order of the autoregressive system
            gnb = 2  # number of global background components (3)
            merge_thr = 0.75  # merging threshold, max correlation allowed (0.86)
            rf = 25  # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
            stride_cnmf = 10  # amount of overlap between the patches in pixels (20)
            K = 23  # number of components per patch (10)
            gSig = [5, 5]  # expected half-size of neurons in pixels [X, Y] (has to be int, not float!)
            method_init = 'greedy_roi'  # initialization method (if analyzing dendritic data using 'sparse_nmf')
            ssub = 2  # spatial subsampling during initialization
            tsub = 2  # temporal subsampling during intialization

            # evaluation parameters
            min_SNR = 7  # signal to noise ratio for accepting a component (default 2)
            SNR_lowest = 3
            rval_thr = 0.85  # space correlation threshold for accepting a component (default 0.85)
            rval_lowest = -1
            cnn_thr = 0.95  # threshold for CNN based classifier (default 0.99)
            cnn_lowest = 0.15  # neurons with cnn probability lower than this value are rejected (default 0.1)

            opts_dict = {'fnames': None, 'fr': fr, 'decay_time': decay_time, 'dxy': dxy, 'nb': gnb, 'rf': rf,
                         'K': K,
                         'gSig': gSig, 'stride': stride_cnmf, 'method_init': method_init, 'rolling_sum': True,
                         'merge_thr': merge_thr, 'only_init': True, 'ssub': ssub, 'tsub': tsub,
                         'SNR_lowest': SNR_lowest, 'cnn_lowest': cnn_lowest, 'min_SNR': min_SNR,
                         'min_cnn_thr': cnn_thr,
                         'rval_lowest': rval_lowest, 'rval_thr': rval_thr, 'use_cnn': True}

            cnm_params = cnmf.params.CNMFParams(params_dict=opts_dict)

        elif mouse == 'M40':
            # dataset dependent parameters
            fr = 30  # imaging rate in frames per second
            decay_time = 0.4  # length of a typical transient in seconds (0.4)
            dxy = (1.66, 1.52)  # spatial resolution (um per pixel) [(1.66, 1.52) for 1x, (0.83, 0.76) for 2x]

            # extraction parameters
            p = 1  # order of the autoregressive system
            gnb = 2  # number of global background components (3)
            merge_thr = 0.75  # merging threshold, max correlation allowed (0.86)
            rf = 25  # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
            stride_cnmf = 10  # amount of overlap between the patches in pixels (20)
            K = 23  # number of components per patch (10)
            gSig = [5, 5]  # expected half-size of neurons in pixels [X, Y] (has to be int, not float!)
            method_init = 'greedy_roi'  # initialization method (if analyzing dendritic data using 'sparse_nmf')
            ssub = 2  # spatial subsampling during initialization
            tsub = 2  # temporal subsampling during intialization

            # evaluation parameters
            min_SNR = 8  # signal to noise ratio for accepting a component (default 2)
            SNR_lowest = 5
            rval_thr = 0.85  # space correlation threshold for accepting a component (default 0.85)
            rval_lowest = -1
            cnn_thr = 0.95  # threshold for CNN based classifier (default 0.99)
            cnn_lowest = 0.18  # neurons with cnn probability lower than this value are rejected (default 0.1)

            opts_dict = {'fnames': None, 'fr': fr, 'decay_time': decay_time, 'dxy': dxy, 'nb': gnb, 'rf': rf,
                         'K': K,
                         'gSig': gSig, 'stride': stride_cnmf, 'method_init': method_init, 'rolling_sum': True,
                         'merge_thr': merge_thr, 'only_init': True, 'ssub': ssub, 'tsub': tsub,
                         'SNR_lowest': SNR_lowest, 'cnn_lowest': cnn_lowest, 'min_SNR': min_SNR,
                         'min_cnn_thr': cnn_thr,
                         'rval_lowest': rval_lowest, 'rval_thr': rval_thr, 'use_cnn': True}
            cnm_params = cnmf.params.CNMFParams(params_dict=opts_dict)

        elif mouse == 'M41':
            # dataset dependent parameters
            fr = 30  # imaging rate in frames per second
            decay_time = 0.4  # length of a typical transient in seconds (0.4)
            dxy = (1.66, 1.52)  # spatial resolution (um per pixel) [(1.66, 1.52) for 1x, (0.83, 0.76) for 2x]

            # extraction parameters
            p = 1  # order of the autoregressive system
            gnb = 2  # number of global background components (3)
            merge_thr = 0.75  # merging threshold, max correlation allowed (0.86)
            rf = 25  # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
            stride_cnmf = 10  # amount of overlap between the patches in pixels (20)
            K = 23  # number of components per patch (10)
            gSig = [5, 5]  # expected half-size of neurons in pixels [X, Y] (has to be int, not float!)
            method_init = 'greedy_roi'  # initialization method (if analyzing dendritic data using 'sparse_nmf')
            ssub = 2  # spatial subsampling during initialization
            tsub = 2  # temporal subsampling during intialization

            # evaluation parameters
            min_SNR = 8  # signal to noise ratio for accepting a component (default 2)
            SNR_lowest = 4.1
            rval_thr = 0.85  # space correlation threshold for accepting a component (default 0.85)
            rval_lowest = -1
            cnn_thr = 0.9  # threshold for CNN based classifier (default 0.99)
            cnn_lowest = 0.22  # neurons with cnn probability lower than this value are rejected (default 0.1)

            opts_dict = {'fnames': None, 'fr': fr, 'decay_time': decay_time, 'dxy': dxy, 'nb': gnb, 'rf': rf,
                         'K': K,
                         'gSig': gSig, 'stride': stride_cnmf, 'method_init': method_init, 'rolling_sum': True,
                         'merge_thr': merge_thr, 'only_init': True, 'ssub': ssub, 'tsub': tsub,
                         'SNR_lowest': SNR_lowest, 'cnn_lowest': cnn_lowest, 'min_SNR': min_SNR,
                         'min_cnn_thr': cnn_thr,
                         'rval_lowest': rval_lowest, 'rval_thr': rval_thr, 'use_cnn': True}

            cnm_params = cnmf.params.CNMFParams(params_dict=opts_dict)

        else:
            return None

        if curr_mouse == 'M40':
            # Set parameters
            pcf_params = {'root': curr_root,  # main directory of this session
                          'trans_length': 0.5,  # minimum length in seconds of a significant transient
                          'trans_thresh': 4,  # factor of sigma above which a transient is significant
                          'bin_length': 2.125,
                          # length in cm VR distance in which to bin dF/F trace (must be divisor of track_length)
                          'bin_window_avg': 3,  # sliding window of bins (left and right) for trace smoothing
                          'bin_base': 0.25,  # fraction of lowest bins that are averaged for baseline calculation
                          'place_thresh': 0.25,  # threshold of being considered for place fields, calculated
                          #     from difference between max and baseline dF/F
                          'min_pf_size': 15,  # minimum size in cm for a place field (should be 15-20 cm)
                          'fluo_infield': 7,
                          # factor above which the mean DF/F in the place field should lie vs. outside the field
                          'trans_time': 0.2,  # fraction of the (unbinned!) signal while the mouse is located in
                          # the place field that should consist of significant transients
                          'track_length': 170,  # length in cm of the virtual reality corridor
                          'split_size': 50}  # size in frames of bootstrapping segments
        else:
            pcf_params = {'root': curr_root,  # main directory of this session
                          'trans_length': 0.5,  # minimum length in seconds of a significant transient
                          'trans_thresh': 4,  # factor of sigma above which a transient is significant
                          'bin_length': 5,
                          # length in cm VR distance in which to bin dF/F trace (must be divisor of track_length)
                          'bin_window_avg': 3,  # sliding window of bins (left and right) for trace smoothing
                          'bin_base': 0.25,  # fraction of lowest bins that are averaged for baseline calculation
                          'place_thresh': 0.25,  # threshold of being considered for place fields, calculated
                          #     from difference between max and baseline dF/F
                          'min_pf_size': 15,  # minimum size in cm for a place field (should be 15-20 cm)
                          'fluo_infield': 7,
                          # factor above which the mean DF/F in the place field should lie vs. outside the field
                          'trans_time': 0.15,  # fraction of the (unbinned!) signal while the mouse is located in
                          # the place field that should consist of significant transients
                          'track_length': 400,  # length in cm of the virtual reality corridor
                          'split_size': 50}  # size in frames of bootstrapping segments

        return cnm_params, pcf_params

    def source_extraction():
        # # Run source extraction
        cnm = run_source_extraction(images, cnm_params, dview=dview)

        # Load local correlation image (should have been created during motion correction)
        try:
            cnm.estimates.Cn = io.imread(curr_root + r'\local_correlation_image.tif')
        except FileNotFoundError:
            save_local_correlation(images, curr_root)
            cnm.estimates.Cn = io.imread(curr_root + r'\local_correlation_image.tif')

        # Plot and save contours of all components
        cnm.estimates.plot_contours(img=cnm.estimates.Cn, display_numbers=False)
        plt.tight_layout()
        fig = plt.gcf()
        fig.set_size_inches((10, 10))
        plt.savefig(os.path.join(curr_root, 'pre_sel_components.png'))
        plt.close()
        save_cnmf(cnm, path=os.path.join(curr_root, 'cnm_pre_selection.hdf5'), verbose=False, overwrite=True)
        return cnm

    def evaluation(cnm):
        # Perform evaluation
        cnm = run_evaluation(images, cnm, dview=dview)

        # Select components, which keeps the data of accepted components and deletes the data of rejected ones
        cnm.estimates.select_components(use_object=True)

        # Detrend calcium data (compute dF/F)
        cnm.params.data['dff_window'] = 2000
        cnm.estimates.detrend_df_f(quantileMin=8, frames_window=cnm.params.data['dff_window'])

        # Save complete CNMF results
        save_cnmf(cnm, path=os.path.join(curr_root, 'cnm_results.hdf5'), overwrite=False, verbose=False)

        # Plot contours of all accepted components
        cnm.estimates.plot_contours(img=cnm.estimates.Cn, display_numbers=False)
        plt.tight_layout()
        fig = plt.gcf()
        fig.set_size_inches((10, 10))
        plt.savefig(os.path.join(curr_root, 'components.png'))
        plt.close()

        return cnm

    def pcf_pipeline(cnm):
        # Initialize PCF object with the raw data (CNM object) and the parameter dict
        pcf = pc.PlaceCellFinder(cnm, pcf_params)
        # If necessary, perform Peters spike prediction
        pcf.cnmf.estimates.spikes = predict_spikes(pcf.cnmf.estimates.F_dff)

        # split traces into trials'
        pcf.split_traces_into_trials()

        pcf.import_behavior_and_align_traces()
        pcf.params['resting_removed'] = True
        pcf.bin_activity_to_vr(remove_resting=pcf.params['resting_removed'])

        # # create significant-transient-only traces
        pcf.create_transient_only_traces()

        pcf.params['trans_time'] = 0.15
        pcf.find_place_cells()

        # Plot place cells
        pcf.plot_all_place_cells(save=True, show_neuron_id=True)

        pcf.save()

    green_start = '\033[1;32;49m'
    green_end = '\033[0;39;49m'

    full_pipe_files = []
    eval_pcf_file = []
    pcf_files = []

    # SEARCH THROUGH FILES AND PERFORM NECESSARY PROCESSING STEPS
    print('Searching for unprocessed sessions in the directory...')
    for step in os.walk(root):
        mmap_file = glob(step[0]+'\\memmap__*.mmap')
        pre_sel_file = glob(step[0] + '\\cnm_pre_selection.hdf5')
        results_file = glob(step[0] + '\\cnm_results.hdf5')
        pcf_file = glob(step[0] + '\\pcf_result*')

        if len(mmap_file) == 1 and len(pre_sel_file) + len(results_file) + len(pcf_file) == 0:
            full_pipe_files.append(mmap_file[0])

        elif len(pre_sel_file) == 1 and len(results_file) + len(pcf_file) == 0:
            eval_pcf_file.append(pre_sel_file[0])

        elif len(results_file) == 1 and len(pcf_file) == 0:
            pcf_files.append(results_file[0])

        elif len(pcf_file):
            pass

    print(f'Found {len(full_pipe_files)} sessions for extraction, evaluation and PCF analysis:')
    print(*full_pipe_files, sep='\n')
    print(f'\nFound {len(eval_pcf_file)} sessions for evaluation and PCF analysis:')
    print(*eval_pcf_file, sep='\n')
    print(f'\nFound {len(pcf_files)} sessions for PCF analysis:')
    print(*pcf_files, sep='\n')
    print('\n\n')

    for file in full_pipe_files:
        # Run source extraction, evaluation and pcf analysis
        curr_mouse = file[57:60]
        curr_root = os.path.dirname(file)
        try:
            cnm_params, pcf_params = set_params(curr_mouse)
        except TypeError:
            continue

        c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)

        mmap_filepath, images = load_mmap(curr_root)  # Load memmap file
        cnm_source = source_extraction()  # Perform source extraction
        cnm_eval = evaluation(cnm_source)  # Perform evaluation
        if curr_mouse != 'M37':
            pcf_pipeline(cnm_eval)  # Perform PCF pipeline

        cm.stop_server(dview=dview)

    for file in eval_pcf_file:
        # Run evaluation and pcf analysis
        curr_mouse = file[57:60]
        curr_root = os.path.dirname(file)
        try:
            cnm_params, pcf_params = set_params(curr_mouse)
        except TypeError:
            continue

        print(green_start + f'\n\nPerforming evaluation and PCF analysis for \n\t {curr_root}.\n' + green_end)

        c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)

        mmap_filepath, images = load_mmap(curr_root)  # Load memmap file
        cnm_source = load_cnmf(curr_root, cnm_filename=os.path.basename(file))
        cnm_eval = evaluation(cnm_source)  # Perform evaluation
        if curr_mouse != 'M37':
            pcf_pipeline(cnm_eval)  # Perform PCF pipeline

        cm.stop_server(dview=dview)

    for file in pcf_files:
        # Run pcf analysis
        curr_mouse = file[57:60]
        curr_root = os.path.dirname(file)
        if curr_mouse != 'M37':
            try:
                cnm_params, pcf_params = set_params(curr_mouse)
            except TypeError:
                continue

            print(green_start + f'\n\nPerforming PCF analysis for \n\t {curr_root}.\n' + green_end)

            cnm_eval = load_cnmf(curr_root, cnm_filename=os.path.basename(file))
            pcf_pipeline(cnm_eval)  # Perform PCF pipeline