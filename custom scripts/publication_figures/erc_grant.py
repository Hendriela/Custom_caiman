#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 25/11/2021 16:44
@author: hheise

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, ndimage
import pandas as pd
import seaborn as sns
from statsmodels.stats import multitest

from schema import common_img, hheise_behav, hheise_placecell


### REWARD CELL ###
key = dict(username='hheise', mouse_id=93, day='2021-07-21', session_num=1)

# Get dF/F traces
traces = (common_img.Segmentation & key).get_traces()

# Split traces into trials
trial_mask = (hheise_placecell.PCAnalysis & key).fetch1('trial_mask')
dff = []
for trial in np.unique(trial_mask):
    dff.append(traces[:, trial_mask == trial])

# Get behavior
behavior = (hheise_behav.VRTrial & key).get_arrays()

# Align traces to 2 sec before and 4 sec after the valve opening and average
before = 1
after = 4

fr = int(np.round((common_img.ScanInfo & key).fetch1('fr')))
bef_frames = int(np.round(before * fr))
aft_frames = int(np.round(after * fr))
win_length = bef_frames + aft_frames

valve_align = []
for behav, trace in zip(behavior, dff):
    # Find valve openings
    valve_openings = np.where(behav[:,5] == 1)[0]
    if len(valve_openings) > 0:
        frame_idx = np.where(behav[:,3] == 1)[0]
        # Find nearest frames
        valve_frames = np.zeros((len(valve_openings), len(trace), win_length))
        for i, valve_idx in enumerate(valve_openings):
            # Get index of nearest frame to valve opening in behavioral array
            glob_idx = frame_idx[np.abs(frame_idx - valve_idx).argmin()]
            # Get number of frame at that index (sum of frame column above glob_idx)
            trace_idx = np.sum(behav[:glob_idx,3], dtype=int)

            valve_frames[i] = trace[:, trace_idx-bef_frames:trace_idx+aft_frames]

        valve_align.append(valve_frames)

valve_aligned = np.concatenate(valve_align)

# Average over all valve openings in this session
mean_valve_aligned = np.mean(valve_aligned, axis=0)

# Find cells where the activity after the valve opening (at index bef_frames+1) is significantly higher than before

# For each neuron, perform a t-test between the samples before and after all valve openings
p_val = np.zeros(mean_valve_aligned.shape[0])*np.nan
for i in range(mean_valve_aligned.shape[0]):
    bef = mean_valve_aligned[i,:bef_frames]
    aft = mean_valve_aligned[i,bef_frames:]
    t, p = stats.ttest_ind(bef, aft, alternative='less')
    p_val[i] = p

# Perform multiple correction against number of cells
p_sig, p_corr, _, _ = multitest.multipletests(p_val, method='bonferroni')

min_p = np.argmin(p_corr)

p_inds = p_corr.argsort()
p_corr_sort = p_corr[p_inds]
mean_valve_aligned_sort = mean_valve_aligned[p_inds]

plt.figure()
plt.plot(mean_valve_aligned[min_p, :])
plt.axvline(bef_frames)

valve_aligned_sort = valve_aligned[:, p_inds, :]

# Neuron with 2nd highest p-value looks best
ex_neuron = valve_aligned_sort[:, 1, :]
# Smooth with Gaussian filter
ex_neuron_smooth = ndimage.gaussian_filter1d(ex_neuron, sigma=1, axis=1)
x = np.tile(np.linspace(-before, after, valve_aligned_sort.shape[2]), valve_aligned_sort.shape[0])

np.savetxt(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Presentations\ERC Grant\reward_cell.txt', ex_neuron_smooth.T, delimiter='\t', fmt='%.4f')
np.savetxt(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Presentations\ERC Grant\reward_cell_x.txt', x[:150].T, delimiter='\t', fmt='%.4f')

data = pd.DataFrame(dict(signal=ex_neuron_smooth.flatten(), timepoint=x))
plt.figure()
sns.lineplot(x='timepoint', y='signal', data=data)
plt.axvline(0, c='r')


### SPEED CELL ###




