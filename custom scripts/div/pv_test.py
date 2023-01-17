#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 06/12/2022 12:58
@author: hheise

"""
import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tif
from scipy import ndimage

fname = r'C:\Users\hheise\Desktop\M2_pv_test\complete_session.tif'

# Index of first flicker and first non-flicker frames, per cycle
flicker_idx = [(1819, 3630),
               (8984, 10794),
               (16154, 17964),
               (23345, 25134),
               (30515, 32304),
               (37685, 39495)]

# First column and row to be used as FOV, ignore upper and left pixels
fov = (100, 70)
stack = tif.imread(fname)

# Slice stack for the large ROI
mov = stack[:, fov[0]:, fov[1]:]

# Take frame-wise average
flat_mov = mov.reshape(40000, -1)
frame_avg = np.mean(flat_mov, axis=1)

# Plot frame-wise avg with flicker borders
plt.figure()
for cycle in flicker_idx:
    plt.axvspan(cycle[0], cycle[1], color='green', alpha=0.3)
plt.plot(frame_avg)

# Correct each flicker period by the brightness difference between
# the last 15 non-flicker and first 15 flicker frame
avg_window = 10
frame_avg_corr = frame_avg.copy()
for cycle in flicker_idx:
    diff = np.mean(frame_avg[cycle[0]:cycle[0]+avg_window]) - np.mean(frame_avg[cycle[0]-1-avg_window:cycle[0]-1])
    frame_avg_corr[cycle[0]:cycle[1]] -= diff
plt.plot(frame_avg_corr)

# Smooth trace
frame_avg_smooth = ndimage.gaussian_filter1d(frame_avg, avg_window)
frame_avg_corr_smooth = ndimage.gaussian_filter1d(frame_avg_corr, avg_window)

plt.figure()
for cycle in flicker_idx:
    plt.axvspan(cycle[0], cycle[1], color='green', alpha=0.3)
plt.plot(frame_avg_smooth)
plt.plot(frame_avg_corr_smooth)

# Calculate stats for Prism export
last_cycle = 0
baseline_off = np.mean(frame_avg_smooth[:1819])
baseline_on = np.mean(frame_avg_smooth[1819:3630])
for idx, cycle in enumerate(flicker_idx):
    print(f'Cycle {idx+1}:')
    curr_off = frame_avg_smooth[last_cycle:cycle[0]]
    curr_on = frame_avg_smooth[cycle[0]:cycle[1]]
    off_mean = np.mean(curr_off-baseline_off)
    on_mean = np.mean(curr_on-baseline_on)
    off_std = np.std(curr_off-baseline_off)
    on_std = np.std(curr_on-baseline_on)
    last_cycle = cycle[1]
    print('\tOFF: {:.4f} +/- {:.4f} (n={})'.format(off_mean, off_std, len(curr_off)))
    print('\tON : {:.4f} +/- {:.4f} (n={})'.format(on_mean, on_std, len(curr_on)))





