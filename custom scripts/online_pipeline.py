#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import numpy as np
import os
import logging
import copy
import matplotlib.pyplot as plt
import random

try:
    if __IPYTHON__:
        # this is used for debugging purposes only.
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

import caiman as cm
from caiman.source_extraction import cnmf as cnmf

import sys
sys.path.append(r'C:\Users\hheise\PycharmProjects\Caiman\custom scripts')
import post_analysis as post

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

cnm = cnmf.cnmf.load_CNMF(r'E:\PhD\Data\CA1\Maus 3 13.08.2019\online_motioncorr_results.hdf5')

# post analysis parameters
# lengths in frames of every trial in this session
frame_list = [5625, 1801, 855, 2397, 9295, 476, 3713, 1816, 903, 4747, 1500, 5024]  # TODO: make it automatic
transient_length = 0.5      # minimum length in seconds of a significant transient
trans_thresh = 4            # factor of sigma above which a signal is considered part of a significant transient
trial_list = np.linspace(11,22,12,dtype='int')  # trial numbers of files that are used for analysis TODO: eliminate hard coded variable
n_bins = 100                # number of bins in which the calcium traces to put

# split traces of neurons into trials
n_trials = len(frame_list)
n_neuron = cnm.estimates.F_dff.shape[0]
session = list(np.zeros(n_neuron))

for neuron in range(n_neuron):
    curr_neur = list(np.zeros(n_trials))       # initialize neuron-list
    session_trace = cnm.estimates.F_dff[neuron]    # temp-save DF/F session trace of this neuron

    for trial in range(n_trials):
        # extract trace of the current trial from the whole session
        if len(session_trace) > frame_list[trial]:
            trial_trace, session_trace = session_trace[:frame_list[trial]], session_trace[frame_list[trial]:]
            curr_neur[trial] = trial_trace  # save trial trace in this neuron's list
        elif len(session_trace) == frame_list[trial]:
            curr_neur[trial] = session_trace
        else:
            print('error')
    session[neuron] = curr_neur                # save data from this neuron to the big session list

# transform session traces into significant-transient-only traces for later place cell screening
session_trans = copy.deepcopy(session)
for neuron in range(len(session)):
    curr_neuron = session[neuron]
    for i in range(len(curr_neuron)):
        trial = curr_neuron[i]
        # get noise level of the data via FWHM
        sigma = post.get_noise_fwhm(trial)
        # get time points where the signal is more than 4x sigma (Koay et al., 2019)
        if sigma == 0:
            idx = []
        else:
            idx = np.where(trial >= trans_thresh*sigma)[0]
        # find blocks of >500 ms length
        blocks = np.split(idx, np.where(np.diff(idx) != 1)[0]+1)
        duration = int(transient_length/(1/cnm.params.data['fr']))
        try:
            transient_idx = np.concatenate([x for x in blocks if x.size >= duration])
        except ValueError:
            transient_idx = []
        # create a transient-only trace of the raw calcium trace
        trans_only = trial.copy()
        select = np.in1d(range(trans_only.shape[0]), transient_idx)
        trans_only[~select] = 0

        # add the transient only trace to the list
        session_trans[neuron][i] = trans_only

#%% import VR data (track position)

data = []
for trial in trial_list:
    data.append(np.loadtxt(f'E:\PhD\Data\CA1\Maus 3 13.03.2019 behavior\{trial}\merged_vr_licks.txt')) #TODO remove hard coding

bin_frame_count = np.zeros((n_bins, n_trials), 'int')
for trial in range(len(data)):  # go through vr data of every trial and prepare it for analysis

    # get a normalized time stamp that is easier to work with (optional)
    data[trial] = np.insert(data[trial], 1, data[trial][:, 0]-data[trial][0, 0], 1)
    # time stamps at data[:,1], position at data[:,2], velocity at data[:,7]

    # bin data in distance chunks
    fr = cnm.params.data['fr']
    bin_borders = np.linspace(-10, 110, n_bins+1)
    idx = np.digitize(data[trial][:, 2], bin_borders)  # get indices of bins

    # create estimate time stamps for frames
    last_stamp = 0
    for i in range(data[trial].shape[0]):
        if last_stamp+(1/fr) < data[trial][i, 1] or i == 0:
            data[trial][i, 5] = 1
            last_stamp = data[trial][i, 1]

    # check how many frames are in each bin
    for i in range(n_bins):
        bin_frame_count[i, trial] = np.sum(data[trial][np.where(idx == i+1), 5])


# Average dF/F for each neuron for each trial for each bin
# goes through every trial, extracts frames according to current bin size, averages it and puts it into
# the data structure "bin_activity", a list of neurons, with every neuron having an array of shape
# (n_trials X n_bins) containing the average dF/F activity of this bin of that trial
bin_activity = list(np.zeros(n_neuron))
for neuron in range(n_neuron):
    curr_neur_act = np.zeros((n_trials, n_bins))
    for trial in range(n_trials):
        curr_trace = session[neuron][trial]
        curr_bins = bin_frame_count[:, trial]
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
bin_avg_activity = np.zeros((n_neuron, n_bins))
for neuron in range(n_neuron):
    bin_avg_activity[neuron] = np.mean(bin_activity[neuron], axis=0)

# place field identification

#%%
plt.figure()
for i in range(test.shape[0]):
    plt.plot(test[i])

# plot single neurons for all trials


# class which contains data for each session. TODO: bring data structure in order (still useful?)
class Session(object):
    def __init__(self):
        t = 0

#%% baseline reduction (should happen in CaImAn dF/F function already)

time_window = 15
frame_window = int(time_window/(1/30))

for i in range(len(test_C)):
    # get the frame windows around the current time point i
    if i < frame_window: curr_left_win = test_C[:i]
    else:  curr_left_win = test_C[i-frame_window:i]
    if i + frame_window > len(test_C):  curr_right_win = test_C[i:]
    else:  curr_right_win = test_C[i:i+frame_window]
    curr_win = np.concatenate((curr_left_win, curr_right_win))

    # get 8th percentile, subtract it from the current time point and paste it into the new array
    pctl = np.percentile(curr_win,8)
    test_base[i] = test_C[i] - pctl



#%% place cell detection
test_neuron = 22
test_trace = bin_avg_activity[test_neuron]
test_trace[10:20] = 0.02
bin_window = 3  # window of bins (to each side) over which the trace should be averaged
bin_base = 0.25 # fraction of lowest bins that should be averaged for baseline calculation
place_thresh = 0.25 # threshold of being considered for place fields, from difference between max and baseline DF/F

# smooth bins by averaging over 3 neighbors
smooth_trace = test_trace.copy()
for i in range(len(test_trace)):
    # get the frame windows around the current time point i
    if i < bin_window: curr_left_bin = test_trace[:i]
    else:  curr_left_bin = test_trace[i-bin_window:i]
    if i + bin_window > len(test_trace):  curr_right_bin = test_trace[i:]
    else:  curr_right_bin = test_trace[i:i+bin_window]
    curr_bin = np.concatenate((curr_left_bin, curr_right_bin))

    smooth_trace[i] = np.mean(curr_bin)

# pre-screen for potential place fields
f_max = max(smooth_trace)   # get maximum DF/F value
f_base = np.mean(np.sort(smooth_trace)[:int(smooth_trace.size*bin_base)])   # get baseline DF/F value
f_thresh = (f_max - f_base) * place_thresh  # get threshold value above which a point is considered part of place field
pot_place_idx = np.where(smooth_trace >= f_thresh)[0]
pot_place_blocks = np.split(pot_place_idx, np.where(np.diff(pot_place_idx) != 1)[0]+1)

#plt.plot(smooth_trace); plt.hlines(f_base,0,n_bins)  # visualize smooth trace and baseline value

# apply place field criteria

min_bin_size = 10  # minimum size in bins a place field should have to be considered as such
fluo_infield = 7   # factor above which the mean DF/F in the place field should lie compared to outside the field
trans_time = 0.2   # relative fraction of the (unbinned!) signal during the time the mouse is spending in the
                   # place field that should be comprised of significant transients
n_splits = 10      # segments the binned DF/F should be split into for bootstrapping. Has to be a divisor of n_bins
# Todo: maybe randomise n_splits? Is it smart to have different split lengths

true_place_fields = []

for pot_place in pot_place_blocks:

    crit_bin_size = False
    crit_infield = False
    crit_trans = False

    pot_place_idx = np.in1d(range(smooth_trace.shape[0]), pot_place)  # get an idx mask for the potential place field
    # check if the place field is at least X bins wide
    if pot_place.size >= min_bin_size:
        crit_bin_size = True
    # check if the mean DF/F inside the field is at least Y times larger than the DF/F outside the field
    if np.mean(smooth_trace[pot_place_idx]) >= fluo_infield*np.mean(smooth_trace[~pot_place_idx]):
        crit_infield = True

    place_frames_trace = []     # stores the trace of all trials when the mouse was in a place field as one data row
    for trial in range(bin_frame_count.shape[1]):
        # get the start and end frame for the current place field from the bin_frame_count array that stores how many
        # frames were pooled for each bin
        curr_place_frames = (np.sum(bin_frame_count[:pot_place[0], trial]),
                             np.sum(bin_frame_count[:pot_place[-1]+1, trial]))
        # attach the transient-only trace in the place field during this trial to the array
        # TODO: not working with the current behavior data, make it work for behavior&frame-trigger data
        place_frames_trace.append(session_trans[test_neuron][trial][curr_place_frames[0]:curr_place_frames[1]+1])
    # one big 1D array that includes all frames where the mouse was located in the place field
    # as this is the transient-only trace, we make it boolean, with False = no transient and True = transient
    place_frames_trace = np.hstack(place_frames_trace).astype('bool')
    # check if at least X percent of the frames are part of a significant transient
    if np.sum(place_frames_trace) >= trans_time*place_frames_trace.shape[0]:
        crit_trans = True

    # if all criteria have been passed, perform boot strapping on the binned DF/F-position trace (see Dombeck2010)
    if crit_bin_size and crit_infield and crit_trans:
        split_trace = np.split(test_trace, n_splits)
        p_counter = 0
        for i in range(1000):
            curr_shuffle = random.sample(split_trace, len(split_trace))
            # TODO implement place-cell finding function
            if place_cell:
                p_counter += 1
        if p_counter/1000 < 0.05:
            true_place_fields.append(pot_place)




