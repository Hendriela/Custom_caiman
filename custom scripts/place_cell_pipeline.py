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


def set_file_paths():
    root = tk.Tk()
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)
    root_dir = filedialog.askdirectory(title='Select folder that contains all trial folders')
    root.withdraw()
    print(f'Root directory:\n {root_dir}')
    return root_dir


def save_cnmf(cnm, root=None, path=None):
    if root is None and path is not None:
        save_path = path
    elif root is not None and path is None:
        save_path = os.path.join(root, 'cnm_results.hdf5')
    else:
        print('Give either a root directory OR a complete path! CNM save directoy ambiguous.')
        return
    if os.path.isfile(save_path):
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


def load_cnmf(root):
    cnm_filename = 'cnm_results.hdf5'
    cnm_filepath = os.path.join(root, cnm_filename)
    cnm_file = glob(cnm_filepath)
    if len(cnm_file) < 1:
        print(f'No file with the name file found in {root}.')
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
        print(f'Found more than one mmap file in {root}. Movie could not be loaded.')
    elif len(mmap_file) < 1:
        print(f'No mmap file found in {root}.')
    else:
        print(f'Loading file {mmap_file[0]}...')
        Yr, dims, T = cm.load_memmap(mmap_file[0])
        images = np.reshape(Yr.T, [T] + list(dims), order='F')
    return mmap_file[0], images


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
    return cnm.refit(images, dview=dview)


def motion_correction(root, params, dview, remove_f_order=True):
    """
    Wrapper function that performs motion correction, saves it as C-order files and can immediately remove F-order files
    to save disk space. Function automatically finds sessions and performs correction on whole sessions separately.
    :param root: str; path in which imaging sessions are searched (files should be in separate trial folders)
    :param params: cnm.params object that holds all parameters necessary for motion correction
    :param remove_f_order: bool flag whether F-order files should be removed to save disk space
    :return mmap_list: list that includes paths of mmap files for all processed sessions
    """
    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split('(\d+)', text)]

    # First, get a list of all folders that include contiguous imaging sessions (have to be motion corrected together)
    dir_list = []
    for step in os.walk(root):
        if len(glob(step[0] + r'\\file*.tif')) > 0 and len(glob(step[0] + r'\\*.mmap')) == 0:
            up_dir = step[0].rsplit(os.sep, 1)[0]
            if up_dir not in dir_list:
                dir_list.append(up_dir)   # this makes a list of all folders that contain single-trial imaging folders

    mmap_list = []
    if len(dir_list) > 0:

        if remove_f_order:
            print('F-order files will be removed after processing.')

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
            mc = MotionCorrect(file_list, dview=dview, **params.get_group('motion'))
            mc.motion_correct(save_movie=True)
            border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0
            # memory map the file in order 'C'
            print(f'Finished motion correction. Starting to save files in C-order...')
            fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C', border_to_0=border_to_0)  # exclude borders
            mmap_list.append(fname_new)

            if remove_f_order:
                for file in mc.fname_tot_els:
                    #print(f'Removing file {file}...')
                    os.remove(file)
            print('Finished!')

    else:
        print('Found no sessions to motion correct!')

    return mmap_list, dview


def save_pcf(pcf):
    save_path = os.path.join(pcf.params['root'], 'pcf_results')
    if os.path.isfile(save_path):
        answer = None
        while answer not in ("y", "n", 'yes', 'no'):
            answer = input(f"File [...]{save_path[-40:]} already exists!\nOverwrite? [y/n] ")
            if answer == "yes" or answer == 'y':
                print('Saving...')
                with open(save_path, 'wb') as file:
                    pickle.dump(pcf, file)
                print(f'PCF results successfully saved at {save_path}')
                return save_path
            elif answer == "no" or answer == 'n':
                print('Saving cancelled.')
                return None
            else:
                print("Please enter yes or no.")


def load_pcf(path):
    with open(path, 'rb') as file:
        obj = pickle.load(file)
    return obj
