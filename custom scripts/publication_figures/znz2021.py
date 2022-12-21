#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 13/09/2021 10:59
@author: hheise

Code to reproduce the data and figures for the ZNZ 2021 poster.
"""

import standard_pipeline.performance_check as performance
import pandas as pd
from datetime import date as dat
import numpy as np
from matplotlib import pyplot as plt

#%% Figure 3A (VR performance)

# Microsphere stroke dates:
#   Batch 3:
#       M33, M38, M39, M41: 20200825
#   Batch 5:
#       M63: 20210302
#       M69: 20210308
#   Batch 7:
#       M85, M86, M90: 20210716
#       M83, M89, M91, M93, M94, M95: 20210721

# Load performance data
path = [r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3']
batch3 = performance.load_performance_data(roots=path,norm_date='20200824', ignore=['M32','M34','M35','M36','M37','M40'])

path = [r'E:\Batch5']
batch5 = performance.load_performance_data(roots=path,norm_date='20210302', ignore=['M62','M65','M67','M68'])

path = [r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch7']
batch7 = performance.load_performance_data(roots=path,norm_date='20210719', ignore=['M81','M82','M87','M92'])

# Filter for relevant dates
batch3_filt = batch3.loc[batch3['sess_norm'] > -7]
M63_filt = batch5.loc[(batch5['sess_norm'] > -5) & (batch5['mouse'] == 'M63')]
M69_filt = batch5.loc[(batch5['sess_norm'] > 0) & (batch5['mouse'] == 'M69')]
batch7_filt1 = batch7.loc[(batch7['sess_norm'] > -10) & (batch7['mouse'].isin(['M85', 'M86', 'M90']))]
batch7_filt2 = batch7.loc[(batch7['sess_norm'] > -5) & (~batch7['mouse'].isin(['M85', 'M86', 'M90']))]
data = pd.concat((batch3_filt, M63_filt, M69_filt, batch7_filt1, batch7_filt2))

# Get mean performance per mouse on each day
def get_mean_performance(df):
    """
    Take the filtered dataframe and calculate mean performance per mouse on each day.
    :param df:
    :return:
    """
    result = []
    i = 0
    for mouse in df.mouse.unique():
        i = 0
        # Get subset of dataset for each mouse
        curr_mouse = df.loc[df['mouse'] == mouse]
        for date in curr_mouse.session_date.unique():
            # For each mouse, get a subset of every session
            curr_sess = curr_mouse.loc[curr_mouse['session_date'] == date]

            # Get the distance in days from stroke surgery (data is sorted to that there are 5 prestroke sessions
            if i < 5:
                norm_day = i - 5
                stroke_day = dat(int(date[:4]), int(date[4:6]), int(date[6:]))
            else:
                norm_day = (dat(int(date[:4]), int(date[4:6]), int(date[6:])) - stroke_day).days

            # Store the data in a pd.Series which is concatenated to one DF in the end
            result.append(pd.DataFrame({"mouse": mouse, "date": date, "performance": curr_sess["licking_binned"].mean(),
                                        "norm_day": norm_day}, index=[i]))
            i += 1
    return pd.concat(result)

mean_performance = get_mean_performance(data)
mean_performance.to_csv(r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Posters\ZNZ Symposium 2021\vr_performance.txt")

#%% Performance Histograms
# Example sessions from M89: 20210625, 20210721, 20210727, 20210808 (full recovery)
# Example sessions from M63: 20210214, 20210302, 20210306, 20210317 (no recovery)

# Set session paths
paths = [r"F:\Batch5\M63\20210214",
         r"F:\Batch5\M63\20210302",
         r"F:\Batch5\M63\20210306",
         r"F:\Batch5\M63\20210317"]

# np.savetxt(r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Posters\ZNZ Symposium 2021\lick_histogram_20210317.txt", out[0], fmt="%.5f")

import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"

# Create figure
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(29.4, 12.8))

# Plot histograms in subplots
performance.plot_lick_histogram_in_figure(path=paths[0], ax=axes[0], label_axes=False, rz_color='green')
performance.plot_lick_histogram_in_figure(path=paths[1], ax=axes[1], label_axes=False, rz_color='green')
performance.plot_lick_histogram_in_figure(path=paths[2], ax=axes[2], label_axes=False, rz_color='green')
performance.plot_lick_histogram_in_figure(path=paths[3], ax=axes[3], label_axes=False, rz_color='green')

# Fix formatting
title_fontsize = 30
y_tick_labelsize = 28

plt.subplots_adjust(hspace=0.5)

axes[0].set_title("Naïve", fontsize=title_fontsize, weight='bold')
axes[1].set_title("Expert", fontsize=title_fontsize, weight='bold')
axes[2].set_title("3 days post microlesions", fontsize=title_fontsize, weight='bold')
axes[3].set_title("21 days post microlesions", fontsize=title_fontsize, weight='bold')

axes[0].tick_params(axis='y', which='major', labelsize=y_tick_labelsize)
axes[1].tick_params(axis='y', which='major', labelsize=y_tick_labelsize)
axes[2].tick_params(axis='y', which='major', labelsize=y_tick_labelsize)
axes[3].tick_params(axis='y', which='major', labelsize=y_tick_labelsize)

axes[0].set_xticks([])
axes[1].set_xticks([])
axes[2].set_xticks([])

axes[3].set_xlabel("VR corridor position bin [cm]", fontsize=36, weight='bold')

axes[3].tick_params(axis='x', which='major', labelsize=y_tick_labelsize)
axes[3].set_xticklabels(axes[3].get_xticks().astype(int), weight='bold')

props = {'ha': 'center', 'va': 'center', 'rotation': 90, 'fontsize': 40}
axes[1].text(-25, 0, "Licks per position bin [%]", props, fontweight='bold')

axes[3].text(32, 82, "RZ", color='green', fontsize=y_tick_labelsize, fontweight='bold')
axes[3].text(139, 82, "RZ", color='green', fontsize=y_tick_labelsize, fontweight='bold')
axes[3].text(245, 82, "RZ", color='green', fontsize=y_tick_labelsize, fontweight='bold')
axes[3].text(353, 82, "RZ", color='green', fontsize=y_tick_labelsize, fontweight='bold')

# Make axis lines thicker
for ax in axes:
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(4)
    # increase tick width
    ax.tick_params(width=4)
    ax.set_yticklabels(ax.get_yticks(), weight='bold')

plt.savefig(r'W:\Neurophysiology-Storage1\Wahl\various_demos\example_VR_behavior.png')
# plt.tight_layout()


#%% Correlation Performance - Place Cells (only batch 3 is analyzed for now)

# Load simple data and filter for the dates around microsphere injection
simple_data = pd.read_pickle(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\simple_data.pickle')
simple_data_filt = simple_data[simple_data.mouse.isin(['M33', 'M38', 'M39', 'M41'])]
simple_data_filt = simple_data_filt[simple_data_filt.session.between(20200818, 20200911)]

# Export data as a TSV
simple_data_filt.to_csv(r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Posters\ZNZ Symposium 2021\neuron_performance_correlation.csv",
                        columns=('mouse', 'session', 'ratio', 'mean_spikerate', 'licking_binned'), index=False)
