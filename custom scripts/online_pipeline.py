#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import numpy as np
import os
import logging
import matplotlib.pyplot as plt

try:
    if __IPYTHON__:
        # this is used for debugging purposes only.
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

import caiman as cm
from caiman.source_extraction import cnmf as cnmf

#%% Set up parameters

# onACID specific parameters
init_batch = 200  # number of frames for initialization (presumably from the first file)
K = 10  # initial number of components
epochs = 2  # number of passes over the data
show_movie = False  # show the movie as the data gets processed
ds_factor = 1  # spatial downsampling factor (increases speed but may lose some fine structure)
sniper_mode = True  # use a CNN to detect new neurons (o/w space correlation)

fr = 30                         # imaging rate in frames per second
decay_time = 3                # length of a typical transient in seconds
dxy = (1, 1)                    # spatial resolution in x and y in (um per pixel)
max_shift_um = (12., 12.)       # maximum shift in um
patch_motion_um = (25., 25.)    # patch size for non-rigid correction in um

# motion correction parameters
pw_rigid = True                 # flag to select rigid vs pw_rigid motion correction
# maximum allowed rigid shift in pixels
max_shifts = [int(a / b) for a, b in zip(max_shift_um, dxy)]
# start a new patch for pw-rigid motion correction every x pixels
strides_mc = tuple([int(a / b) for a, b in zip(patch_motion_um, dxy)])
overlaps = (12, 12)             # overlap between patches (size of patch in pixels: strides+overlaps)
max_deviation_rigid = 3         # maximum deviation allowed for patch with respect to rigid shifts

# CNMF parameters
p = 1                       # order of the autoregressive system
gnb = 3                     # number of global background components
merge_thr = 0.86            # merging threshold, max correlation allowed
rf = 50                     # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
stride_cnmf = 20            # amount of overlap between the patches in pixels
K = 10                       # number of components per patch
gSig = [13, 11]             # expected half size of neurons in pixels
method_init = 'greedy_roi'  # initialization method
ssub = 2                    # spatial subsampling during initialization
tsub = 2                    # temporal subsampling during intialization
gSig = tuple(np.ceil(np.array(gSig) / ds_factor).astype('int'))  # recompute gSig if downsampling is involved

# Evaluation parameters
# signal to noise ratio for accepting a component (default 0.5 and 2)
SNR_lowest = 3
min_SNR = 6
# space correlation threshold for accepting a component (default -1 and 0.85)
rval_lowest = -1
rval_thr = 0.75
# threshold for CNN based classifier (default 0.1 and 0.99)
cnn_lowest = 0.1
cnn_thr = 0.99
use_cnn = True

# set up cluster for parallel processing
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

# enter parameters in the dictionary
opts_dict = {'fnames': fnames, 'fr': fr,  'decay_time': decay_time, 'dxy': dxy, 'pw_rigid': pw_rigid,
             'max_shifts': max_shifts, 'strides': strides_mc, 'overlaps': overlaps,
             'max_deviation_rigid': max_deviation_rigid, 'border_nan': 'copy', 'nb': gnb, 'rf': rf, 'K': K, 'gSig': gSig,
             'stride': stride_cnmf, 'method_init': method_init, 'rolling_sum': True, 'merge_thr': merge_thr,
             'n_processes': n_processes,  'only_init': True, 'ssub': ssub, 'tsub': tsub, 'min_SNR': min_SNR,
             'rval_thr': rval_thr, 'min_cnn_thr': cnn_thr, 'cnn_lowest': cnn_lowest, 'use_cnn': True}

#%% Motion correction:
# motion correction can be either performed before, then onACID uses the finished C-order mmap file
# otherwise, onACID can perform motion correction on the fly

# TODO: implement/copy from demo script

#%% onACID

# run the online analysis pipeline
cnm = cnmf.online_cnmf.OnACID(params=opts)
cnm.fit_online()
# load mmap images and create correlation image
Yr, dims, T = cm.load_memmap(fnames[0])
images = np.reshape(Yr.T, [T] + list(dims), order='F')
Cn = cm.local_correlations(images, swap_dim=False)
cnm.estimates.Cn = Cn
# evaluate components (again)
cnm.estimates.evaluate_components(images, cnm.params, dview=dview)
# save results
cnm.save(os.path.splitext(fnames[0])[0] + '_results.hdf5')

#%% post analysis

cnm = cnmf.cnmf.load_CNMF(r'E:\PhD\Data\CA1\Maus 3 13.08.2019\memmap__d1_512_d2_512_d3_1_order_C_frames_38152__results.hdf5')

# split traces of neurons into trials
frame_list = [5625, 1801, 855, 2397, 9295, 476, 3713, 1816, 903, 4747, 1500, 5024]  # TODO: make it automatic

n_trials = len(frame_list)
n_neuron = cnm.estimates.F_dff.shape[0]
session = list(np.zeros(n_neuron))

for neuron in range(n_neuron):
    curr_neur = list(np.zeros(n_trials))       # initialize neuron-list
    session_trace = cnm.estimates.F_dff[neuron]    # temp-save complete session trace of this neuron
    for trial in range(n_trials):
        # extract trace of the current trial from the whole session
        if len(session_trace) > frame_list[trial]:
            trial_trace, session_trace = session_trace[:frame_list[trial]], session_trace[frame_list[trial]:]
            curr_neur[trial] = trial_trace         # save trial trace in this neuron's list
        elif len(session_trace) == frame_list[trial]:
            curr_neur[trial] = session_trace
        else:
            print('error')
    session[neuron] = curr_neur                # save data from this neuron to the big session list

# import VR data (track position)
trial_list = np.linspace(11,22,12,dtype='int')  #TODO: eliminate hard coded variable
n_bins = 100  # TODO: eliminate hard coded variable

data = []
for trial in trial_list:
    data.append(np.loadtxt(f'E:\PhD\Data\CA1\Maus 3 13.03.2019 behavior\{trial}\merged_vr_licks.txt'))

bin_frame_count = np.zeros((n_bins,n_trials),'int')
for trial in range(len(data)):  # go through vr data of every trial and prepare it for analysis

    # get a normalized time stamp that is easier to work with (optional)
    data[trial] = np.insert(data[trial],1,data[trial][:,0]-data[trial][0,0],1)
    # time stamps at data[:,1], position at data[:,2], velocity at data[:,7]

    # bin data in distance chunks
    fr = cnm.params.data['fr']
    bin_borders = np.linspace(-10,110,n_bins+1)
    idx = np.digitize(data[trial][:,2], bin_borders)  # get indices of bins

    # create estimate time stamps for frames
    last_stamp = 0
    for i in range(data[trial].shape[0]):
        if last_stamp+(1/fr) < data[trial][i,1] or i == 0:
            data[trial][i,5] = 1
            last_stamp = data[trial][i,1]

    # check how many frames are in each bin
    for i in range(n_bins):
        bin_frame_count[i,trial] = np.sum(data[trial][np.where(idx==i+1),5])


# Average dF/F for each neuron for each trial for each bin
# goes through every trial, extracts frames according to current bin size, averages it and puts it into
# the data structure "bin_activity", a list of neurons, with every neuron having an array of shape
# (n_trials X n_bins) containing the average dF/F activity of this bin of that trial
bin_activity = list(np.zeros(n_neuron))
for neuron in range(n_neuron):
    curr_neur_act = np.zeros((n_trials,n_bins))
    for trial in range(n_trials):
        curr_trace = session[neuron][trial]
        curr_bins = bin_frame_count[:,trial]
        curr_act_bin = np.zeros(n_bins)
        for bin_no in range(n_bins):
            # extract the trace of the current bin from the trial trace
            if len(curr_trace) > curr_bins[bin_no]:
                trace_to_avg, curr_trace = curr_trace[:curr_bins[bin_no]], curr_trace[curr_bins[bin_no]:]
            elif len(curr_trace) == curr_bins[bin_no]:
                trace_to_avg = curr_trace
            curr_act_bin[bin_no] = np.mean(trace_to_avg)
        curr_neur_act[trial] = curr_act_bin
    bin_activity[neuron] = curr_neur_act

# Get average activity across trials of every neuron for every bin
bin_avg_activity = np.zeros((n_neuron,n_bins))
for neuron in range(n_neuron):
    bin_avg_activity[neuron] = np.mean(bin_activity[neuron], axis=0)


plt.figure()
for i in range(test.shape[0]):
    plt.plot(test[i])

# plot single neurons for all trials


# class which contains data for each session. TODO: bring data structure in order
class Session(object):
    def __init__(self):
        t = 0






