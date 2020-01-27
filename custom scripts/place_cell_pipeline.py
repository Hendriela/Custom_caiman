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

#%% File and directory handling


def set_file_paths():
    root = tk.Tk()
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)
    root_dir = filedialog.askdirectory(title='Select folder that contains all trial folders')
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


def load_cnmf(root, cnm_filename='cnm_results.hdf5'):
    cnm_filepath = os.path.join(root, cnm_filename)
    cnm_file = glob(cnm_filepath)
    if len(cnm_file) < 1:
        raise FileNotFoundError(f'No file with the name {cnm_filepath} found.')
    else:
        print(f'Loading file {cnm_file[0]}...')
        return cnmf.load_CNMF(cnm_file[0])


def load_mmap(root):
    """
    Loads the motion corrected mmap file of the whole session for CaImAn procedure.
    :param root: folder that holds the session mmap file (should hold only one mmap file)
    :return images: memory-mapped array of the whole session (format [n_frame x X x Y])
    :return mmap_file[0]: path to the loaded mmap file
    """
    mmap_file = glob(root + r'\\*.mmap')
    if len(mmap_file) > 1:
        raise FileNotFoundError(f'Found more than one mmap file in {root}. Movie could not be loaded.')
    elif len(mmap_file) < 1:
        raise FileNotFoundError(f'No mmap file found in {root}.')
    else:
        print(f'Loading file {mmap_file[0]}...')
        Yr, dims, T = cm.load_memmap(mmap_file[0])
        images = np.reshape(Yr.T, [T] + list(dims), order='F')
        return mmap_file[0], images


def load_pcf(root):
    pcf_path = glob(root + r'\\pcf_results.pickle')
    if len(pcf_path) < 1:
        pcf_path = glob(root + r'\\pcf_results')
        if len(pcf_path) < 1:
            raise FileNotFoundError(f'No pcf file found in {root}.')
        elif len(pcf_path) > 1:
            raise FileNotFoundError(f'More than one pcf file found in {root}.')
    elif len(pcf_path) > 1:
        raise FileNotFoundError(f'More than one pcf file found in {root}.')
    with open(pcf_path[0], 'rb') as file:
        obj = pickle.load(file)
    return obj

#%% CNMF wrapper functions

def whole_caiman_pipeline_mouse(root, cnm_params, pcf_params, dview, make_lcm=True, network='all', overwrite=False):

    for step in os.walk(root):
        if len([s for s in step[2] if 'memmap__d1_' in s]) == 1:    # is there a session memory map?
            if len([s for s in step[2] if 'place_cells' in s]) == 0 or overwrite:    # has it already been processed?
                if network == 'all' or network == step[0][-2:]:     # is it the correct network?
                    whole_caiman_pipeline_session(step[0], cnm_params, pcf_params, dview, make_lcm)

def whole_caiman_pipeline_session(root, cnm_params, pcf_params, dview, make_lcm=False, save_pre_sel_img=True, overwrite=False):
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
    cnm_params = cnm_params.change_params({'fnames': mmap_file})
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
    cnm.estimates.select_components(use_object=True)
    print('\tFinished, now creating dF/F trace...')
    cnm.estimates.detrend_df_f(quantileMin=8, frames_window=500)
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
    # create significant-transient-only traces
    pcf.create_transient_only_traces()
    # align the frames to the VR position using merged behavioral data
    pcf.save()
    pcf.import_behavior_and_align_traces()
    # look for place cells
    pcf.find_place_cells()
    # save pcf object
    if len(pcf.place_cells) > 0:
        pcf.plot_all_place_cells(save=True, show_neuron_id=True)
    pcf.save(overwrite=True)
    # delete cnmf object if pcf object could be saved successfully (cnmf is included in pcf)
    os.remove(os.path.join(root, 'cnm_pre_selection.hdf5'))
    print('Finished!')

def get_local_correlation(images):
    """
    Calculates local correlation map of a movie.
    :param images:  mmap file of the movie with the dimensions [n_frames x X x Y]
    :return lcm: local correlation map
    """
    lcm = cm.local_correlations(images, swap_dim=False)
    lcm[np.isnan(lcm)] = 0
    return lcm


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
    params = params.change_params({'p': 0})
    cnm = cnmf.CNMF(params.get('patch', 'n_processes'), params=params, dview=dview)
    cnm = cnm.fit(images)
    cnm.params.change_params({'p': params.get('temporal', 'p')})
    cnm2 = cnm.refit(images, dview=dview)
    cnm2.estimates.dims = (512, 512)
    return cnm2

#%% Motion correction wrapper functions


def motion_correction(root, params, dview, remove_f_order=True, remove_c_order=False):
    """
    Wrapper function that performs motion correction, saves it as C-order files and can immediately remove F-order files
    to save disk space. Function automatically finds sessions and performs correction on whole sessions separately.
    :param root: str; path in which imaging sessions are searched (files should be in separate trial folders)
    :param params: cnm.params object that holds all parameters necessary for motion correction
    :param remove_f_order: bool flag whether F-order files should be removed to save disk space
    :param remove_c_order: bool flag whether single-trial C-order files should be removed to save disk space
    :return mmap_list: list that includes paths of mmap files for all processed sessions
    """
    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split('(\d+)', text)]

    # First, get a list of all folders that include contiguous imaging sessions (have to be motion corrected together)
    dir_list = []
    for step in os.walk(root):
        if len(glob(step[0] + r'\\file*.tif')) > 0:
            up_dir = step[0].rsplit(os.sep, 1)[0]
            if len(glob(up_dir + r'\\memmap__d1_*.mmap')) == 0 and up_dir not in dir_list:
                dir_list.append(up_dir)   # this makes a list of all folders that contain single-trial imaging folders

    mmap_list = []
    if len(dir_list) > 0:

        if remove_f_order:
            print('\nF-order files will be removed after processing.')
        if remove_c_order:
            print('Single-trial C-order files will be removed after processing.')

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
            print(f'\nNow starting to motion correct session {session} ({len(file_list)} trials).')
            # perform motion correction

            ### FOR M25 ###
            if session[-2:] == r'N3':
                params.change_params({'dxy': (0.83, 0.76)})
            else:
                params.change_params({'dxy': (1.66, 1.52)})
            ###
            mc = MotionCorrect(file_list, dview=dview, **params.get_group('motion'))
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
            print('Finished!')

    else:
        print('Found no sessions to motion correct!')

    return mmap_list, dview


