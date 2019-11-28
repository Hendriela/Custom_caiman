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
import re
import pickle


def set_file_paths():
    root = tk.Tk()
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)
    root_dir = filedialog.askdirectory(title='Select folder that contains all trial folders')
    root.withdraw()
    print(f'Root directory:\n {root_dir}')
    return root_dir


def motion_correction(root, params, remove_f_order=True):

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

            # create cluster
            try:    #TODO: make it work in case a cluster is already set up
                c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)
            except Exception:
                pass

            # list of all .tif files of that session which should be corrected together, sorted by their trial number
            file_list = glob(session + r'\\*\\file*.tif')
            file_list.sort(key=natural_keys)
            print(f'\nNow starting to motion correct session {session} ({len(file_list)} trials).')
            # perform motion correction
            mc = MotionCorrect(file_list, dview=dview, **params.get_group('motion'))
            mc.motion_correct(save_movie=True)
            border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0
            # memory map the file in order 'C'
            fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C', border_to_0=border_to_0)  # exclude borders
            mmap_list.append(fname_new)

            if remove_f_order:
                for file in mc.fname_tot_els:
                    #print(f'Removing file {file}...')
                    os.remove(file)
            print('Finished!')

            # stop cluster
            cm.stop_server(dview=dview)
    else:
        print('Found no sessions to motion correct!')

    return mmap_list


def save_pcf(path, obj):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)


def load_pcf(path):
    with open(path, 'rb') as file:
        obj = pickle.load(file)
    return obj
