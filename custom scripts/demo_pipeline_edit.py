#!/usr/bin/env python

"""
Complete demo pipeline for processing two photon calcium imaging data using the
CaImAn batch algorithm. The processing pipeline included motion correction,
source extraction and deconvolution. The demo shows how to construct the
params, MotionCorrect and cnmf objects and call the relevant functions. You
can also run a large part of the pipeline with a single method (cnmf.fit_file)
See inside for details.

Demo is also available as a jupyter notebook (see demo_pipeline.ipynb)
Dataset couresy of Sue Ann Koay and David Tank (Princeton University)

This demo pertains to two photon data. For a complete analysis pipeline for
one photon microendoscopic data see demo_pipeline_cnmfE.py

copyright GNU General Public License v2.0
authors: @agiovann and @epnev
"""

import cv2
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(r'C:\Users\hheise\PycharmProjects\Caiman\custom scripts')
#import caiman_helper as helper
import post_analysis as post
import imageio

try:
    cv2.setNumThreads(0)
except:
    pass

try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.utils import download_demo

#%%
# Set up the logger (optional); change this if you like.
# You can log to a file using the filename parameter, or make the output more
# or less verbose by setting level to logging.DEBUG, logging.INFO,
# logging.WARNING, or logging.ERROR

logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.ERROR)

#%%
def main():
    pass  # For compatibility between running under Spyder and the CLI

#%% Select file(s) to be processed (download if not present)
    root = '/Users/hheiser/Desktop/testing data/chronic_M2N3/0d_baseline/channel1'

    fname_list = [r'C:\Users\hheise\caiman_data\PhD data\Maus 3 13.08.2019\file_00011.tif',
              r'C:\Users\hheise\caiman_data\PhD data\Maus 3 13.08.2019\file_00012.tif',
              r'C:\Users\hheise\caiman_data\PhD data\Maus 3 13.08.2019\file_00013.tif',
              r'C:\Users\hheise\caiman_data\PhD data\Maus 3 13.08.2019\file_00014.tif',
              r'C:\Users\hheise\caiman_data\PhD data\Maus 3 13.08.2019\file_00015.tif',
              r'C:\Users\hheise\caiman_data\PhD data\Maus 3 13.08.2019\file_00016.tif',
              r'C:\Users\hheise\caiman_data\PhD data\Maus 3 13.08.2019\file_00017.tif',
              r'C:\Users\hheise\caiman_data\PhD data\Maus 3 13.08.2019\file_00018.tif',
              r'C:\Users\hheise\caiman_data\PhD data\Maus 3 13.08.2019\file_00019.tif',
              r'C:\Users\hheise\caiman_data\PhD data\Maus 3 13.08.2019\file_00020.tif',
              r'C:\Users\hheise\caiman_data\PhD data\Maus 3 13.08.2019\file_00021.tif',
              r'C:\Users\hheise\caiman_data\PhD data\Maus 3 13.08.2019\file_00022.tif']

    fnames = [r'E:\PhD\Data\CA1\Maus 3 13.08.2019\file_00012.tif']
    #fnames = ['Sue_2x_3000_40_-46.tif']  # filename to be processed
    if fnames[0] in ['Sue_2x_3000_40_-46.tif', 'demoMovie.tif']:
        fnames = [download_demo(fnames[0])]

#%% First setup some parameters for data and motion correction

    # dataset dependent parameters
    fr = 30             # imaging rate in frames per second
    decay_time = 3    # length of a typical transient in seconds (0.4)
    dxy = (1, 1)      # spatial resolution in x and y in (um per pixel)
    # note the lower than usual spatial resolution here
    max_shift_um = (12., 12.)       # maximum shift in um
    patch_motion_um = (25., 25.)  # patch size for non-rigid correction in um

    # motion correction parameters
    pw_rigid = True       # flag to select rigid vs pw_rigid motion correction
    # maximum allowed rigid shift in pixels
    max_shifts = [int(a/b) for a, b in zip(max_shift_um, dxy)]
    # start a new patch for pw-rigid motion correction every x pixels
    strides = tuple([int(a/b) for a, b in zip(patch_motion_um, dxy)])
    # overlap between pathes (size of patch in pixels: strides+overlaps)
    overlaps = (12, 12)
    # maximum deviation allowed for patch with respect to rigid shifts
    max_deviation_rigid = 3

    mc_dict = {
        'fnames': fnames,
        'fr': fr,
        'decay_time': decay_time,
        'dxy': dxy,
        'pw_rigid': pw_rigid,
        'max_shifts': max_shifts,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': 'copy'
    }

    opts = params.CNMFParams(params_dict=mc_dict)

#%% play the movie (optional)
    # playing the movie using opencv. It requires loading the movie in memory.
    # To close the video press q
    display_images = True

    if display_images:
        m_orig = cm.load_movie_chain(fnames)
        ds_ratio = 0.2
        moviehandle = m_orig.resize(1, 1, ds_ratio)
        moviehandle.play(q_max=99.5, fr=30, magnification=1, do_loop=True)

#%% start a cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)

#%% MOTION CORRECTION
    # first we create a motion correction object with the specified parameters
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
    # note that the file is not loaded in memory

#%% Run (piecewise-rigid motion) correction using NoRMCorre
    mc.motion_correct(save_movie=True)

#%% compare with original movie
    if display_images:
        m_orig = cm.load_movie_chain(fnames)
        m_els = cm.load(mc.mmap_file)
        ds_ratio = 0.2
        moviehandle = cm.concatenate([m_orig.resize(1, 1, ds_ratio) - mc.min_mov*mc.nonneg_movie,
                                      m_els.resize(1, 1, ds_ratio)], axis=2)
        moviehandle.play(fr=15, q_max=99.5, magnification=2)  # press q to exit

#%% MEMORY MAPPING
    border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0
    # you can include the boundaries of the FOV if you used the 'copy' option
    # during motion correction, although be careful about the components near
    # the boundaries

    fname_new = r'E:\PhD\Data\CA1\Maus 3 13.08.2019\file_00014_d1_512_d2_512_d3_1_order_C_frames_2397_.mmap'

    # memory map the file in order 'C'
    fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C',
                               border_to_0=border_to_0)  # exclude borders

    # now load the file
    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    # load frames in python format (T x X x Y)

#%% restart cluster to clean up memory
    cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)

#%%  parameters for source extraction and deconvolution
    p = 1                       # order of the autoregressive system
    gnb = 3                     # number of global background components (3)
    merge_thr = 0.86            # merging threshold, max correlation allowed (0.86)
    rf = 50
    # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
    stride_cnmf = 20            # amount of overlap between the patches in pixels (20)
    K = 10                      # number of components per patch (10)
    gSig = [13, 11]             # expected half size of neurons in pixels (13,11)
    # initialization method (if analyzing dendritic data using 'sparse_nmf')
    method_init = 'greedy_roi'
    ssub = 2                    # spatial subsampling during initialization
    tsub = 2                    # temporal subsampling during intialization

    # parameters for component evaluation
    opts_dict = {'fnames': fnames,
                 'fr': fr,
                 'nb': gnb,
                 'rf': rf,
                 'K': K,
                 'gSig': gSig,
                 'stride': stride_cnmf,
                 'method_init': method_init,
                 'rolling_sum': True,
                 'merge_thr': merge_thr,
                 'n_processes': n_processes,
                 'only_init': True,
                 'ssub': ssub,
                 'tsub': tsub}

    opts.change_params(params_dict=opts_dict)
#%% RUN CNMF ON PATCHES
    # First extract spatial and temporal components on patches and combine them
    # for this step deconvolution is turned off (p=0)

    #opts.change_params({'p': 1,'rf':None, 'only_init':False})
    opts.change_params({'p': 0})
    cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
    cnm = cnm.fit(images)

#%% RUN CNMF SEEDED WITH MANUAL MASK

    # load mask
    mask = np.asarray(imageio.imread('/Users/hheiser/Desktop/testing data/file_00020_no_motion/avg_mask_fixed.png'), dtype=bool)

    # get component ROIs from the mask and plot them
    Ain, labels, mR = cm.base.rois.extract_binary_masks(mask)

    # plot original mask and extracted labels to check mask
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(-mR,cmap='binary')
    ax[0].set_title('Original mask')
    ax[1].imshow(labels)
    ax[1].set_title('Extracted labelled ROIs')

    """"
    plt.figure()
    crd = cm.utils.visualization.plot_contours(
        Ain.astype('float32'), mR, thr=0.99, display_numbers=True) # todo check if this is important for the pipeline
    plt.title('Contour plots of detected ROIs in the structural channel')
    """

    opts.change_params({'rf': None, 'only_init': False})

    # run CNMF seeded with this mask
    cnm_corr_seed = cnmf.CNMF(n_processes, params=opts, dview=dview, Ain=Ain)
    cnm_corr_seed = cnm_corr_seed.fit(images)
    #cnm_seed = cnm_seed.fit_file(motion_correct=False)

#%% ALTERNATE WAY TO RUN THE PIPELINE AT ONCE
    #   you can also perform the motion correction plus cnmf fitting steps
    #   simultaneously after defining your parameters object using
    #  cnm1 = cnmf.CNMF(n_processes, params=opts, dview=dview)
    #  cnm1.fit_file(motion_correct=True)

#%% plot contours of found components
    Cn = cm.local_correlations(images, swap_dim=False)
    Cn[np.isnan(Cn)] = 0
    cnm.estimates.plot_contours(img=Cn)
    plt.title('Contour plots of found components')

#%% RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
    cnm.params.change_params({'p': p})
    cnm2 = cnm.refit(images, dview=dview)

#%% COMPONENT EVALUATION
    # the components are evaluated in three ways:
    #   a) the shape of each component must be correlated with the data
    #   b) a minimum peak SNR is required over the length of a transient
    #   c) each shape passes a CNN based classifier
    min_SNR = 6  # signal to noise ratio for accepting a component (default 2)
    SNR_lowest = 3
    rval_thr = 0.7 # space correlation threshold for accepting a component (default 0.85)
    cnn_thr = 0.99  # threshold for CNN based classifier (default 0.99)
    cnn_lowest = 0.1 # neurons with cnn probability lower than this value are rejected (default 0.1)

    cnm2.params.set('quality', {'decay_time': decay_time,
                               'SNR_lowest': SNR_lowest,
                               'min_SNR': min_SNR,
                               'rval_thr': rval_thr,
                               'use_cnn': True,
                               'min_cnn_thr': cnn_thr,
                               'cnn_lowest': cnn_lowest})
    cnm2.estimates.evaluate_components(images, params=cnm2.params, dview=dview)

#%% PLOT COMPONENTS
    cnm2.estimates.plot_contours(img=Cn, idx=cnm2.estimates.idx_components)

#%% VIEW TRACES (accepted and rejected)

    if display_images:
        cnm2.estimates.view_components(images, img=Cn,
                                      idx=cnm2.estimates.idx_components)
        cnm2.estimates.view_components(images, img=Cn,
                                      idx=cnm2.estimates.idx_components_bad)
#%% update object with selected components
    #### -> will delete rejected components!
    cnm2.estimates.select_components(use_object=True)
#%% Extract DF/F values
    cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=250)

#%% Show final traces
    cnm2.estimates.view_components(img=Cn)

#%% reconstruct denoised movie (press q to exit)
    if display_images:
        cnm2.estimates.play_movie(images, q_max=99.9, gain_res=2,
                                  magnification=1,
                                  bpx=border_to_0,
                                  include_bck=True)  # background not shown

#%% STOP CLUSTER and clean up log files
    cm.stop_server(dview=dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)

#%% save results

    dirname = fnames[0][:-4] + "_results.hdf5"
    cnm2.estimates.Cn = Cn
    cnm2.save(dirname)

    #load results
    cnm2 = cnmf.load_CNMF(dirname)

    mov_name = fnames[0][:-4] + "_movie_restored_2_gain.tif"
    helper.save_movie(cnm2.estimates,images,mov_name,frame_range=range(200),include_bck=True)

#%%
# This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    main()
