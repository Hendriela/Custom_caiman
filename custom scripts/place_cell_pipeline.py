#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Hendrik Heiser
# created on 22 October 2019

import tkinter as tk
from tkinter import filedialog
import numpy as np
import os
from glob import glob
import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf


def set_file_paths():
    root = tk.Tk()
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)
    root_dir = filedialog.askdirectory(title='Select folder that contains all trial folders')
    root.withdraw()

    folder_list = glob(root_dir+'/*/')  # list of directories for behavioral import
    tif_list = glob(root_dir + r'/*/*.tif')  # list of tif movies, used by CaImAn

    return folder_list, tif_list


def motion_correction(params, remove_f_order=True):
    # create cluster
    if remove_f_order:
        print('F-order files will be removed after processing.')

    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)

    # perform motion correction
    mc = MotionCorrect(params.get('data', 'fnames'), dview=dview, **params.get_group('motion'))
    mc.motion_correct(save_movie=True)
    border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0
    # memory map the file in order 'C'
    fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C', border_to_0=border_to_0)  # exclude borders

    # stop cluster
    cm.stop_server(dview=dview)

    if remove_f_order:
        for file in mc.fname_tot_els:
            print(f'Removing file {file}...')
            os.remove(file)

    return fname_new



