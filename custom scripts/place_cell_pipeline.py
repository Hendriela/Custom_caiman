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
import place_cell_class as pc
from behavior_import import progress
from skimage import io
import preprocess as pre
import tifffile as tif
from datetime import datetime

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
            raise FileNotFoundError(f'No pcf file found in {os.path.join(root, fname)}.')
    else:
        pcf_path = glob(root + r'\\pcf_results_manual.pickle')
        if len(pcf_path) < 1:
            pcf_path = glob(root + r'\\pcf_results.pickle')
            if len(pcf_path) < 1:
                pcf_path = glob(root + r'\\pcf_results')
                if len(pcf_path) < 1:
                    raise FileNotFoundError(f'No pcf file found in {root}.')
            elif len(pcf_path) > 1:
                raise FileNotFoundError(f'More than one pcf file found in {root}.')
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

#%% CNMF wrapper functions


def whole_caiman_pipeline_mouse(root, cnm_params, pcf_params, dview, make_lcm=True, network='all', overwrite=False):

    for step in os.walk(root):
        if len([s for s in step[2] if 'memmap__d1_' in s]) == 1:    # is there a session memory map?
            if len([s for s in step[2] if 'place_cells' in s]) == 0 or overwrite:    # has it already been processed?
                if network == 'all' or network == step[0][-2:]:     # is it the correct network?
                    whole_caiman_pipeline_session(step[0], cnm_params, pcf_params, dview, make_lcm)


def whole_caiman_pipeline_session(root, cnm_params, pcf_params, dview, make_lcm=False, save_pre_sel_img=True,
                                  overwrite=False, only_extraction=False):
    """
    Wrapper for the complete caiman and place cell pipeline. Performs source extraction, evaluation and place cell
    search for one session/network (one mmap file).
    :param root: directory of the session, has to include the complete mmap file of that session
    :param cnm_params: CNMFParams object holding all parameters for CaImAn
    :param pcf_params: dict holding all parameters for the place cell finder
    :param dview: pointer for the CaImAn cluster. Cluster has to be started before calling this function.
    :param make_lcm: bool flag whether a local correlation map should be created and saved together with the cnm object.
    LCM is necessary for plotting, but can take long to compute and runs the system out of memory for files > ~45 GB.
    """
    print(f'\nStarting processing of {root}...')
    mmap_file, images = load_mmap(root)
    cnm_params = cnm_params.change_params({'fnames': mmap_file[0]})
    pcf_params['root'] = root
    cnm = run_source_extraction(images, cnm_params, dview=dview)
    save_cnmf(cnm, path=os.path.join(root, 'cnm_pre_selection.hdf5'), verbose=False, overwrite=overwrite)
    if make_lcm:
        print('\tFinished source extraction, now calculating local correlation map...')
        if images.shape[0] > 40000:
            # skip every 4th index by making a mask
            mask = np.ones(images.shape[0], dtype=bool)
            mask[::4] = 0
            half_images = images[mask, :, :]
            lcm = get_local_correlation(half_images)
        else:
            lcm = get_local_correlation(images)
        cnm.estimates.Cn = lcm
    if save_pre_sel_img:
        cnm.estimates.plot_contours(img=cnm.estimates.Cn, display_numbers=False)
        plt.tight_layout()
        fig = plt.gcf()
        fig.set_size_inches((10, 10))
        plt.savefig(os.path.join(root, 'pre_sel_components.png'))
        plt.close()
    print('\tFinished, now evaluating components...')
    cnm = run_evaluation(images, cnm, dview=dview)
    save_cnmf(cnm, path=os.path.join(root, 'cnm_pre_selection.hdf5'), overwrite=True, verbose=False)
    if not only_extraction:
        cnm.estimates.select_components(use_object=True)
    print('\tFinished, now creating dF/F trace...')
    cnm.estimates.detrend_df_f(quantileMin=8, frames_window=5000)
    cnm.params.data['dff_window'] = 5000
    if not only_extraction:
        save_cnmf(cnm, path=os.path.join(root, 'cnm_results.hdf5'), overwrite=True, verbose=False)
        if make_lcm:
            cnm.estimates.plot_contours(img=cnm.estimates.Cn, display_numbers=False)
            plt.tight_layout()
            fig = plt.gcf()
            fig.set_size_inches((10, 10))
            plt.savefig(os.path.join(root, 'components.png'))
            plt.close()
        print('\tFinished, now searching for place cells...')
        pcf = pc.PlaceCellFinder(cnm, pcf_params)
        # split traces into trials
        pcf.split_traces_into_trials()
        # align the frames to the VR position using merged behavioral data
        pcf.save()
        pcf.import_behavior_and_align_traces(remove_resting_frames=True)
        pcf.params['remove_resting_frames'] = True
        # create significant-transient-only traces
        pcf.create_transient_only_traces()
        pcf.save(overwrite=True)
        # look for place cells
        pcf.find_place_cells()
        # save pcf object
        if len(pcf.place_cells) > 0:
            pcf.plot_all_place_cells(save=False, show_neuron_id=True)
        pcf.save(overwrite=True)
        # delete cnmf object if pcf object could be saved successfully (cnmf is included in pcf)
        print('Finished!')


def get_local_correlation(movie):
    """
    Calculates local correlation map of a movie.
    :param movie:  mmap file of the movie with the dimensions [n_frames x X x Y]
    :return lcm: local correlation map
    """
    lcm = cm.local_correlations(movie, swap_dim=False)
    lcm[np.isnan(lcm)] = 0
    return lcm


def save_local_correlation(movie, path):
    cor = get_local_correlation(movie)
    fname = path + r'\local_correlation_image.tif'
    io.imsave(fname, cor.astype('float32'))
    print(f'Saved local correlation image at {fname}.')
    return fname


def save_average_image(movie, path):
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

    # run refinement of masks and extraction of traces
    params = params.change_params({'p': 1})
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


def motion_correction(root, params, dview, percentile=0.1,
                      remove_f_order=True, remove_c_order=False, get_images=True, overwrite=False):
    """
    Wrapper function that performs motion correction, saves it as C-order files and can immediately remove F-order files
    to save disk space. Function automatically finds sessions and performs correction on whole sessions separately.
    :param root: str; path in which imaging sessions are searched (files should be in separate trial folders)
    :param params: cnm.params object that holds all parameters necessary for motion correction
    :param dview: link to Caimans processing server
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
        if len(glob(step[0] + r'\\file_0*.tif')) > 0:
            up_dir = step[0].rsplit(os.sep, 1)[0]
            if len(glob(up_dir + r'\\memmap__d1_*.mmap')) == 0 or overwrite and up_dir not in dir_list:
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
            file_list = glob(session + r'\\*\\file*.tif')
            file_list.sort(key=natural_keys)
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

                new_path = raw_file[:-4] + '_corrected.tif'  # avoid overwriting
                tif.imwrite(new_path, data=stack)
                temp_files.append(new_path)

            # Perform motion correction

            # ### FOR M25 ###
            # if session[-2:] == r'N3':
            #     params.change_params({'dxy': (0.83, 0.76)})
            # else:
            #     params.change_params({'dxy': (1.66, 1.52)})
            # ###
            mc = MotionCorrect(temp_files, dview=dview, **params.get_group('motion'))
            mc.motion_correct(save_movie=True)
            border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0

            # memory map the file in order 'C'
            print(f'Finished motion correction. Starting to save files in C-order...')
            fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C', border_to_0=border_to_0)
            mmap_list.append(fname_new)

            if remove_f_order:
                for file in mc.fname_tot_els:
                    #print(f'Removing file {file}...')
                    os.remove(file)
            if remove_c_order:
                temp_file = os.path.join(*file_list[0].split('\\')[:-1], 'temp_filenames.txt')
                with open(temp_file, 'r') as temp:
                    lines = temp.readlines()
                    for line in lines:
                        path = line[:-1]
                        os.remove(path)
                os.remove(temp_file)

            # remove temporary corrected.tif files
            for file in temp_files:
                os.remove(file)

            # compute local correlation and mean intensity image
            if get_images:
                print(f'Finished. Now computing local correlation and mean intensity images...')
                Yr, dims, T = cm.load_memmap(fname_new)
                images = np.reshape(Yr.T, [T] + list(dims), order='F')
                out = save_local_correlation(images, session)
                out = save_average_image(images, session)

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

        print(f'Checking component {idx[i]+1}...')
        if idx[i] in cnm.estimates.idx_components:
            print(green_start+f'\nComponent {idx[i]+1} got accepted, all lower threshold were passed!'+green_end+'\n\n\tUpper thresholds:\n')

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
            print(f'\nComponent {idx[i] + 1} did not get accepted. \n\n\tChecking thresholds:\n')

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
                print(red_start+f'Result: Component {idx[i]+1} got rejected because it failed at least one lower threshold!\n\n'+red_end)
            elif upper_thresh_failed == 3 and not lower_thresh_failed:
                print(red_start+f'Result: Component {idx[i]+1} got rejected because it met all lower, but no upper thresholds!\n\n'+red_end)
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
