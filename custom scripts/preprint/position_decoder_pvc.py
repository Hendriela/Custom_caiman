#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 11/04/2023 14:00
@author: hheise

"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from schema import hheise_placecell, common_match, hheise_behav


def plot_pvc_curve(y_vals, session_stdev, bin_size=5, show=False):
    """Plots the pvc curve

        Parameters
        ----------
        y_vals : array-like
            data points of the pvc curve (idx = bin distance)
        bin_size : bool, optional
        show : bool, optional

       Returns
       -------
       fig: figure object
           a figure object of the pvc curve
    """
    fig = plt.figure()
    x_axis = np.arange(0., len(y_vals)* bin_size, bin_size)  # bin size
    plt.errorbar(x_axis, y_vals, session_stdev, figure=fig)
    plt.ylim(bottom=0); plt.ylabel('Mean PVC')
    plt.xlim(left=0); plt.xlabel('Offset Distances (cm)')
    if show:
        plt.show(block=True)
    return fig


def pvc_curve(session1, session2, plot=True, max_delta_bins=30):
    """Calculate the mean pvc curve between two sessions.

        Parameters
        ----------
        activity_matrix : 2D array containing (float, dim1 = bins, dim2 = neurons)
        plot: bool, optional
        max_delta_bins: int, optional
            max difference in bin distance

       Returns
       -------
       curve_yvals:
           array of mean pvc curve (idx = delta_bin)
    """

    # Filter out neurons that are not active in both cells
    neuron_mask = np.sum(np.array([~np.isnan(session1).any(axis=1),
                                   ~np.isnan(session2).any(axis=1)]).astype(int), axis=0) == 2
    session1 = session1[neuron_mask].T
    session2 = session2[neuron_mask].T
    print(f'{np.sum(neuron_mask)} neurons available')

    num_bins = np.size(session1, 0)
    num_neurons = np.size(session1, 1)
    curve_yvals = np.empty(max_delta_bins + 1)
    curve_stdev = np.empty(max_delta_bins + 1)
    for delta_bin in range(max_delta_bins + 1):
        pvc_vals = []
        for offset in range(num_bins - delta_bin):
            idx_x = offset
            idx_y = offset + delta_bin
            pvc_xy_num = pvc_xy_den_term1 = pvc_xy_den_term2 = 0
            for neuron in range(num_neurons):
                pvc_xy_num += session1[idx_x][neuron] * session2[idx_y][neuron]
                pvc_xy_den_term1 += session1[idx_x][neuron] * session1[idx_x][neuron]
                pvc_xy_den_term2 += session2[idx_y][neuron] * session2[idx_y][neuron]
            pvc_xy = pvc_xy_num / (np.sqrt(pvc_xy_den_term1 * pvc_xy_den_term2))
            pvc_vals.append(pvc_xy)
        mean_pvc_delta_bin = np.mean(pvc_vals)
        stdev_delta_bin = np.std(pvc_vals)
        curve_yvals[delta_bin] = mean_pvc_delta_bin
        curve_stdev[delta_bin] = stdev_delta_bin

    if plot:
        plot_pvc_curve(curve_yvals, curve_stdev, show=True)

    return curve_yvals, curve_stdev


def pvc_across_sessions(session1, session2, plot_heatmap=False, plot_in_ax=None, plot_zones=False):
    """
    Compute PVC across sessions (see Shuman2020 Fig 3c).
    For each position bin X, compute population vector correlation of all neurons between two sessions.
    The resulting matrix has position bins of session1 on x-axis, and position bins of session2 on y-axis.
    Mean PVC curves are computed by comparing not the same position bin X between both sessions, but offsetting the
    position bin in one session by N centimeters, and average across all position-bin-pairs with the same offset.
    """

    # Filter out neurons that are not active in both cells
    neuron_mask = np.sum(np.array([~np.isnan(session1).any(axis=1),
                                   ~np.isnan(session2).any(axis=1)]).astype(int), axis=0) == 2
    session1 = session1[neuron_mask]
    session2 = session2[neuron_mask]
    print(f'{np.sum(neuron_mask)} neurons available')

    ### Compute PVC across all positions between the two sessions
    pvc_matrix = np.zeros((session1.shape[1], session2.shape[1])) * np.nan
    for x in range(session1.shape[1]):
        for y in range(session2.shape[1]):
            # Multiply spatial maps of the same neuron of both sessions and sum across neurons (numerator)
            numerator = np.sum(session1[:, x] * session2[:, y])
            # Multiply spatial maps of the same neuron of both sessions with each other, sum across neurons, and multiply (denominator)
            denominator = np.sum(session1[:, x] * session1[:, x]) * np.sum(session2[:, y] * session2[:, y])

            # Compute PVC for this position combination
            pvc_matrix[x, y] = numerator / np.sqrt(denominator)

    if plot_heatmap:
        if plot_in_ax is not None:
            curr_ax = plot_in_ax
        else:
            plt.figure()
            curr_ax = plt.gca()
        sns.heatmap(pvc_matrix, ax=curr_ax, vmin=0, vmax=1, cbar=False, cmap='turbo')
        if plot_zones:
            zone_borders = (hheise_behav.CorridorPattern & 'pattern="training"').rescale_borders(n_bins=80)
            for zone in zone_borders:
                curr_ax.axvspan(zone[0], zone[1], facecolor='green', alpha=0.4)

    return pvc_matrix


# Fetch the spatial activity maps of matched cells
queries = (
           (common_match.MatchedIndex & 'mouse_id=33'),       # 407 cells
           # (common_match.MatchedIndex & 'mouse_id=38'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=41'),   # 246 cells
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=69' & 'day<="2021-03-23"'),     # 350 cells
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=85'),   # 250 cells
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=86'),   # 86 cells
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=90'),   # 131 cells
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=91'),   # 299 cells
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=93'),   # 397 cells
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=108' & 'day<"2022-09-09"'),     # 316 cells
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=114' & 'day<"2022-09-09"'),     # 307 cells
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=115' & 'day<"2022-09-09"'),     # 331 cells
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=122' & 'day<"2022-09-09"'),     # 401 cells
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=121' & 'day<"2022-09-09"'),     # 791 cells
           # (common_match.MatchedIndex & 'mouse_id=110' & 'day<"2022-09-09"'),     # 21 cells
)

spatial_maps = []
for query in queries:
    # PVC matrices with spikerate instead of dFF activity are a bit sharper, improving contrast
    spatial_map = query.get_matched_data(table=hheise_placecell.BinnedActivity.ROI, attribute='bin_spikerate',
                                         extra_restriction=dict(corridor_type=0, place_cell_id=2),
                                         return_array=True, relative_dates=True, surgery='Microsphere injection')
    spatial_maps.append(spatial_map)

# For each network, compute PVC matrices between two sessions
fig, axes = plt.subplots(1, 4, sharex='all', sharey='all')
pvc_pre = pvc_across_sessions(session1=spatial_maps[1]['121_1'][0][:, 1],
                              session2=spatial_maps[1]['121_1'][0][:, 4],
                              plot_heatmap=True, plot_in_ax=axes[0], plot_zones=True)
pvc_pre_post = pvc_across_sessions(session1=spatial_maps[1]['121_1'][0][:, 4],
                                   session2=spatial_maps[1]['121_1'][0][:, 5],
                                   plot_heatmap=True, plot_in_ax=axes[1], plot_zones=True)
pvc_early = pvc_across_sessions(session1=spatial_maps[1]['121_1'][0][:, 6],
                                session2=spatial_maps[1]['121_1'][0][:, 7],
                                plot_heatmap=True, plot_in_ax=axes[2], plot_zones=True)
pvc_late = pvc_across_sessions(session1=spatial_maps[1]['121_1'][0][:, 10],
                               session2=spatial_maps[1]['121_1'][0][:, 11],
                               plot_heatmap=True, plot_in_ax=axes[3], plot_zones=True)

# Make plot pretty
titles = ['Prestroke (-3 -> 0)', 'Pre-Post (0 -> 3)', 'Early Post (6 -> 9)', 'Late Post (18 -> 21)']
for ax, title in zip(axes, titles):
    ax.set_title(title)

### Take average of all pre, early and post pvc matrices
DAY_DIFF = 3    # The day difference between sessions to be compared (usually 3)
# days = np.array(spatial_maps[1]['69_1'][1])
# s_maps = spatial_maps[1]['122_1'][0]
# last_pre_day = days[np.searchsorted(days, 0, side='right')-1]   # Find last prestroke day
vmin = 0
vmax = 1

if vmin is not None and vmax is not None:
    plot_cbar = False
else:
    plot_cbar = True


for mouse in spatial_maps:
    mouse_id = list(mouse.keys())[0]
    if mouse_id not in ('41_1', '69_1', '85_1', '86_1', '90_1', '91_1'):
        continue
    print('Mouse_ID:', mouse_id)
    days = np.array(mouse[mouse_id][1])
    last_pre_day = days[np.searchsorted(days, 0, side='right') - 1]  # Find last prestroke day
    pre = []
    pre_post = []
    early = []
    late = []
    for day_idx, day in enumerate(days):
        next_day_idx = np.where(days == day + DAY_DIFF)[0]

        # After stroke, ignore small differences between sessions (do not have to be 3 days apart, sometimes 2 days)
        # In that case, use the next session irrespective of distance
        if (len(next_day_idx) == 0) and (1 < day < np.max(days)):
            next_day_idx = [day_idx + 1]

        # If a session 3 days later exists, compute the correlation of all cells between these sessions
        # Do not analyze session 1 day after stroke (unreliable data)
        if len(next_day_idx) == 1:
            curr_mat = pvc_across_sessions(session1=mouse[mouse_id][0][:, day_idx],
                                           session2=mouse[mouse_id][0][:, next_day_idx[0]],
                                           plot_heatmap=False)
            print('Day', day, '- next_day', days[next_day_idx[0]])
            if day < last_pre_day:
                pre.append(curr_mat)
                print('\t-> Pre')
            elif day == last_pre_day:
                pre_post.append(curr_mat)
                print('\t-> Pre-Post')
            # elif last_pre_day < days[next_day_idx[0]] <= 6:
            elif last_pre_day < day < 6:
                early.append(curr_mat)
                print('\t-> Early Post')
            elif 6 <= day:
                late.append(curr_mat)
                print('\t-> Late Post')

    avg_pre = np.mean(np.stack(pre), axis=0)
    pre_post = pre_post[0]
    avg_early = np.mean(np.stack(early), axis=0)
    avg_late = np.mean(np.stack(late), axis=0)

    fig, axes = plt.subplots(1, 4, sharex='all', sharey='all', figsize=(18, 8))
    sns.heatmap(avg_pre, ax=axes[0], vmin=vmin, vmax=vmax, cbar=plot_cbar, cmap='turbo')
    sns.heatmap(pre_post, ax=axes[1], vmin=vmin, vmax=vmax, cbar=plot_cbar, cmap='turbo')
    sns.heatmap(avg_early, ax=axes[2], vmin=vmin, vmax=vmax, cbar=plot_cbar, cmap='turbo')
    sns.heatmap(avg_late, ax=axes[3], vmin=vmin, vmax=vmax, cbar=True, cmap='turbo')

    # Make plot pretty
    titles = ['Prestroke', 'Pre-Post', 'Early Post', 'Late Post']
    for ax, title in zip(axes, titles):
        ax.set_title(title)
    zone_borders = (hheise_behav.CorridorPattern & 'pattern="training"').rescale_borders(n_bins=80)
    for curr_ax in axes:
        for zone in zone_borders:
            # curr_ax.axvspan(zone[0], zone[1], facecolor='gray', alpha=0.4)
            curr_ax.axvline(zone[0], color='black', linestyle='--')
            curr_ax.axvline(zone[1], color='black', linestyle='--')
    fig.canvas.manager.set_window_title(mouse_id)
    plt.tight_layout()



### Calculate and plot PVC curves

DAY_DIFF = 3    # The day difference between sessions to be compared (usually 3)
days = np.array(spatial_maps[2]['121_1'][1])
pre = []
pre_post = []
early = []
late = []

for day_idx, day in enumerate(days):
    next_day_idx = np.where(days == day + DAY_DIFF)[0]

    # If a session 3 days later exists, compute the correlation of all cells between these sessions
    # Do not analyze session 1 day after stroke (unreliable data)
    if day + DAY_DIFF != 1 and len(next_day_idx) == 1:
        curr_curve, curr_std = pvc_curve(session1=spatial_maps[2]['121_1'][0][:, day_idx],
                                       session2=spatial_maps[2]['121_1'][0][:, next_day_idx[0]],
                                       plot=False)
        if day < 0:
            pre.append(np.array([curr_curve, curr_std]))
            print('\r -> pre')
        elif day == 0:
            pre_post.append(np.array([curr_curve, curr_std]))
            print('\r -> pre-post')
        elif 0 < day < 9:
            early.append(np.array([curr_curve, curr_std]))
            print('\r -> early')
        elif 9 < day:
            late.append(np.array([curr_curve, curr_std]))
            print('\r -> late')

avg_pre = np.mean(np.stack(pre), axis=0).T
pre_post = pre_post[0].T
avg_early = np.mean(np.stack(early), axis=0).T
avg_late = np.mean(np.stack(late), axis=0).T

# Export for prism
np.savetxt(r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Papers\preprint\figure3\PVC\avg_pre.csv', avg_pre, delimiter=',', fmt='%.6f')
np.savetxt(r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Papers\preprint\figure3\PVC\pre_post.csv', pre_post, delimiter=',', fmt='%.6f')
np.savetxt(r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Papers\preprint\figure3\PVC\avg_early.csv', avg_early, delimiter=',', fmt='%.6f')
np.savetxt(r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Papers\preprint\figure3\PVC\avg_late.csv', avg_late, delimiter=',', fmt='%.6f')

# PVC curves for one day per period
bin_size = 5
fig = plt.figure()

pvc_curve_pre, pvc_curve_pre_std = pvc_curve(session1=spatial_maps[2]['121_1'][0][:, 1],
                                             session2=spatial_maps[2]['121_1'][0][:, 4],
                                             plot=False)
pvc_curve_pre_post, pvc_curve_pre_post_std = pvc_curve(session1=spatial_maps[2]['121_1'][0][:, 4],
                                                       session2=spatial_maps[2]['121_1'][0][:, 5],
                                                       plot=False)
pvc_curve_early, pvc_curve_early_std = pvc_curve(session1=spatial_maps[2]['121_1'][0][:, 6],
                                                 session2=spatial_maps[2]['121_1'][0][:, 7],
                                                 plot=False)
pvc_curve_late, pvc_curve_late_std = pvc_curve(session1=spatial_maps[2]['121_1'][0][:, 10],
                                               session2=spatial_maps[2]['121_1'][0][:, 11],
                                               plot=False)

x_axis = np.arange(0., len(pvc_curve_pre) * bin_size, bin_size)  # bin size
plt.errorbar(x_axis, pvc_curve_pre, pvc_curve_pre_std, figure=fig, label='Pre')
plt.errorbar(x_axis, pvc_curve_pre_post, pvc_curve_pre_post_std, figure=fig, label='Pre-Post')
plt.errorbar(x_axis, pvc_curve_early, pvc_curve_early_std, figure=fig, label='Early')
plt.errorbar(x_axis, pvc_curve_late, pvc_curve_late_std, figure=fig, label='Late')
plt.legend()

plt.ylim(bottom=0)
plt.ylabel('Mean PVC')
plt.xlim(left=0)
plt.xlabel('Offset Distances (cm)')
