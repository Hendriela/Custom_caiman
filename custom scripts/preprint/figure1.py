#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 11/04/2023 13:50
@author: hheise

Finished code to recreate plots for figure 1
"""

import itertools

import matplotlib
matplotlib.rcParams['font.sans-serif'] = "Arial"  # Use same font as Prism
# matplotlib.rcParams['font.family'] = "sans-serif"

from matplotlib import pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'
from typing import Optional, Tuple, List, Iterable
import os
from datetime import timedelta
from scipy.ndimage import gaussian_filter1d
import tifffile as tif
import cv2
import standard_pipeline.performance_check as performance
from itertools import combinations
import pandas as pd
import seaborn as sns
from scipy import stats
import numpy as np
import pickle
from scipy import ndimage


from schema import hheise_behav, common_mice, hheise_placecell, common_img, common_match, hheise_hist, hheise_grouping
from util import helper
from preprint import data_cleaning as dc

mouse_ids = [33, 41,  # Batch 3
             63, 69,  # Batch 5
             83, 85, 86, 89, 90, 91, 93, 94, 95,  # Batch 7
             108, 110, 111, 112, 113, 114, 115, 116, 121, 122]  # Batch 8

no_deficit = [93, 91, 94, 95]
no_deficit_flicker = [111, 114, 116]
recovery = [33, 83, 85, 86, 89, 90, 113]
deficit_no_flicker = [41, 63, 69, 121]
deficit_flicker = [108, 110, 112]
sham_injection = [115, 122]

mice = [*no_deficit, *no_deficit_flicker, *recovery, *deficit_no_flicker, *deficit_flicker, *sham_injection]

grouping = pd.DataFrame(data=dict(mouse_id=mice,
                                  group=[*['no_deficit'] * len(no_deficit),
                                         *['no_deficit_flicker'] * len(no_deficit_flicker),
                                         *['recovery'] * len(recovery),
                                         *['deficit_no_flicker'] * len(deficit_no_flicker),
                                         *['deficit_flicker'] * len(deficit_flicker),
                                         *['sham_injection'] * len(sham_injection)]))
grouping_2 = pd.DataFrame(data=dict(mouse_id=mice,
                                    group=[*['no_deficit'] * len(no_deficit), *['no_deficit'] * len(no_deficit_flicker),
                                           *['recovery'] * len(recovery),
                                           *['deficit_no_flicker'] * len(deficit_no_flicker),
                                           *['deficit_flicker'] * len(deficit_flicker),
                                           *['no_deficit'] * len(sham_injection)]))

folder = r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Papers\preprint'


#%% Figure 1
def example_fov():

    avg_pre = (common_img.QualityControl & 'mouse_id=41' & 'day="2020-08-24"').fetch1('avg_image')
    # tif.imwrite(os.path.join(folder, '41_20200824_fov_overview.tif'), avg)

    avg_post = (common_img.QualityControl & 'mouse_id=41' & 'day="2020-08-26"').fetch1('avg_image')
    # tif.imwrite(os.path.join(folder, '41_20200824_fov_overview_damage.tif'), avg)

    # Example FOV with spheres
    post = cv2.imread(os.path.join(folder, 'figure2', '41_20200826_poststroke_raw.png'))
    ind = np.where(post == 255)
    post[ind[0], ind[1], :] = [0, 0, 255]
    # avg_post_rgb_norm = (255 * (avg_post_rgb - np.min(avg_post_rgb)) / np.ptp(avg_post_rgb)).astype(int)        # Normalize image
    cv2.imwrite(os.path.join(folder, 'figure2', '41_20200826_poststroke_thresh.png'), post)

    cor_post = (common_img.QualityControl & 'mouse_id=41' & 'day="2020-08-26"').fetch1('cor_image')


def example_tracked_cells():

    def fetch_trial_mask(day):
        tm = (hheise_placecell.PCAnalysis & 'mouse_id=41' & f'day="{day}"' & 'place_cell_id=2').fetch1('trial_mask')
        t_borders = np.where(np.diff(tm) == 1)[0]
        return tm, t_borders

    def plot_cells(trace_df, cell_rows):
        # Fetch trial masks to separate trials
        tm1, tb1 = fetch_trial_mask("2020-08-18")
        tm2, tb2 = fetch_trial_mask("2020-08-30")
        tm3, tb3 = fetch_trial_mask("2020-09-11")
        borders = [tb1, tb2, tb3]

        days = np.array([-7, 5, 17])

        fig, ax = plt.subplots(nrows=3, ncols=1, sharex='none', sharey='all')

        for day_idx, day in enumerate(days):
            [ax[day_idx].axvline(x, color='red') for x in borders[day_idx]]
            for cell in cell_rows:
                act = trace_df[day].loc[cell]
                ax[day_idx].plot(act, label=cell)
        ax[-1].legend()


    date1, img1 = (common_img.QualityControl & 'mouse_id=41' & 'day="2020-08-18"').fetch1('day', 'avg_image')
    date2, img2 = (common_img.QualityControl & 'mouse_id=41' & 'day="2020-08-30"').fetch1('day', 'avg_image')
    date3, img3 = (common_img.QualityControl & 'mouse_id=41' & 'day="2020-09-11"').fetch1('day', 'avg_image')

    # Coordinates of same landmark in three images (x, y)
    point1 = (348, 367)
    point2 = (351, 353)
    point3 = (356, 362)

    # Image borders around the given coordinates (left, right, top, bottom)
    borders = (75, 25, 15, 85)

    cut1 = img1[point1[0] - borders[0]: point1[0] + borders[1], point1[1] - borders[2]: point1[1] + borders[3]]

    plt.imshow(cut1, cmap='Greys_r')

    # Get cell IDs of some landmark cells (from cell_matching GUI)
    match_matrix = (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=41').construct_matrix()['41_1']

    with open(r'C:\Users\hheise.UZH\PycharmProjects\Caiman\custom scripts\preprint\Filippo\neural_data\dff.pkl', 'rb') as file:
        dff = pickle.load(file)

    cell_rows = [272, 119, 95]     # row index in matched matrix of matched cell
    cell_rows = [89, 95, 116, 119, 240, 312, 330, 333, 346]        # row index in matched matrix of matched cell

    plot_cells(trace_df=dff[41], cell_rows=[312, 232, 272])
    plot_cells(trace_df=dff[41], cell_rows=cell_rows)

    # Matrix rows of final selected cells for Figure 1C
    selected_cells = match_matrix.loc[[89, 240, 346]]

    # Plot contours of selected cells
    fig, ax = plt.subplots(nrows=1, ncols=3)
    (common_img.Segmentation.ROI & 'mouse_id=41' & 'day="2020-08-18"' &
     f'mask_id in {helper.in_query([367, 790, 895])}').plot_contours(show_id=True, background='avg_image', ax=ax[0])
    (common_img.Segmentation.ROI & 'mouse_id=41' & 'day="2020-08-30"' &
     f'mask_id in {helper.in_query([863, 848, 896])}').plot_contours(show_id=True, background='avg_image', ax=ax[1])
    (common_img.Segmentation.ROI & 'mouse_id=41' & 'day="2020-09-11"' &
     f'mask_id in {helper.in_query([645, 196, 193])}').plot_contours(show_id=True, background='avg_image', ax=ax[2])

    # Plot contours of selected cells
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex='all', sharey='all', layout='constrained')
    (common_img.Segmentation.ROI & 'mouse_id=41' & 'day="2020-08-18"' &
     f'mask_id in {helper.in_query(selected_cells.iloc[:, 0])}').plot_contours(show_id=True, background='avg_image', ax=ax[0])
    (common_img.Segmentation.ROI & 'mouse_id=41' & 'day="2020-08-30"' &
     f'mask_id in {helper.in_query(selected_cells.iloc[:, 6])}').plot_contours(show_id=True, background='avg_image', ax=ax[1])
    (common_img.Segmentation.ROI & 'mouse_id=41' & 'day="2020-09-11"' &
     f'mask_id in {helper.in_query(selected_cells.iloc[:, 10])}').plot_contours(show_id=True, background='avg_image', ax=ax[2])
    ax[0].set_title(selected_cells.columns[0])
    ax[1].set_title(selected_cells.columns[6])
    ax[2].set_title(selected_cells.columns[10])

    ### Plot traces of selected cells in a single trial of each day
    # day1: trial 6
    # day2: trial 16
    # day3: trial 4

    # Fetch trial masks to separate trials
    tm1 = (hheise_placecell.PCAnalysis & 'mouse_id=41' & 'day="2020-08-18"' & 'place_cell_id=2').fetch1('trial_mask')
    tm2 = (hheise_placecell.PCAnalysis & 'mouse_id=41' & 'day="2020-08-30"' & 'place_cell_id=2').fetch1('trial_mask')
    tm3 = (hheise_placecell.PCAnalysis & 'mouse_id=41' & 'day="2020-09-11"' & 'place_cell_id=2').fetch1('trial_mask')
    tms = [tm1, tm2, tm3]
    trials = [6, 16, 4]
    cell_rows = [89, 240, 346]
    days = np.array([-7, 5, 17])

    fig, ax = plt.subplots(nrows=3, ncols=1, sharex='none', sharey='all', layout='constrained')

    for day_idx, day in enumerate(days):
        for cell in cell_rows:
            act = dff[41][day].loc[cell][tms[day_idx] == trials[day_idx]]
            x = np.linspace(0, len(act)/30, len(act))
            ax[day_idx].plot(x, act, label=cell)
    ax[-1].legend()
    fig.savefig(r"C:\Users\hheise.UZH\Desktop\preprint\figure1\tracked_cells_example_traces.svg")


def example_lick_histogram():
    ### Example lick histogram

    # Set session paths (main figure)
    paths = [r"F:\Batch5\M63\20210214",
             r"F:\Batch5\M63\20210302",
             r"F:\Batch5\M63\20210306",
             r"F:\Batch5\M63\20210317"]
    # Set session paths (supplementary figure)
    paths = [r"F:\Batch5\M69\20210216",
             r"F:\Batch5\M69\20210304",
             r"F:\Batch5\M69\20210311",
             r"F:\Batch5\M69\20210323"]

    # Create figure
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(29.4, 12.8))

    # Plot histograms in subplots
    # Todo: implement performance lick histogram in Datajoint Table
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

    plt.savefig(os.path.join(folder, 'example_VR_behavior_supp.svg'), transparent=True)


def learning_curves():
    # Get performance for X sessions before the first microsphere injection for each mouse
    dfs = []
    for mouse in mouse_ids:
        # Get day of surgery
        surgery_day = (common_mice.Surgery() & 'surgery_type="Microsphere injection"' & f'mouse_id={mouse}').fetch(
            'surgery_date')[0].date()
        # Get date and performance for each session before the surgery day
        days = (hheise_behav.VRPerformance & 'username="hheise"' & f'mouse_id={mouse}' & f'day<="{surgery_day}"').fetch('day')
        perf = (hheise_behav.VRPerformance & 'username="hheise"' & f'mouse_id={mouse}' & f'day<="{surgery_day}"').get_mean('binned_lick_ratio')

        # Transform dates into days before surgery
        rel_days = [(d - surgery_day).days for d in days]
        # Get session number before surgery
        rel_sess = np.arange(-len(perf), 0)

        dfs.append(pd.DataFrame(dict(mouse_id=mouse, days=days, rel_days=rel_days, rel_sess=rel_sess, perf=perf)))
    df = pd.concat(dfs, ignore_index=True)

    # sns.lineplot(data=df_all, x='rel_sess', y='perf')

    # Export for prism
    df_exp = df.pivot(index='rel_sess', columns='mouse_id', values='perf')
    df_exp.to_csv(os.path.join(folder, 'figure1', 'learning_curve1.csv'))


def place_cell_plot():
    # Get primary keys of all place cells in the normal corridor, apply strict quality criteria
    thresh = (common_img.Segmentation.ROI * hheise_placecell.SpatialInformation.ROI & 'is_pc=1' & 'accepted=1').fetch(
        'pf_threshold')
    thresh_quant = np.quantile(thresh, 0.95)
    restrictions = dict(is_pc=1, accepted=1, p_si=0, p_stability=0, corridor_type=0)
    pk_pc_si = (common_img.Segmentation.ROI * hheise_placecell.SpatialInformation.ROI & restrictions &
                'snr>10' & 'r>0.9' & 'cnn>0.9' & f'pf_threshold>{thresh_quant}').fetch('KEY')

    # Get place cells from Bartos criteria
    restrictions = dict(is_place_cell=1, accepted=1, corridor_type=0, place_cell_id=2)
    pk_pc = (common_img.Segmentation.ROI * hheise_placecell.PlaceCell.ROI & restrictions &
             'snr>10' & 'r>0.9' & 'cnn>0.9').fetch('KEY')

    # Get spatial activity maps
    act = (hheise_placecell.BinnedActivity.ROI() & pk_pc).fetch('bin_activity')
    # Only keep traces from normal corridor trials
    norm_act_list = [curr_act[:, (hheise_behav.VRSession & cell_pk).get_normal_trials()]
                     for curr_act, cell_pk in zip(act, pk_pc)]
    # norm_act = np.array(norm_act_list1, dtype='object')
    fields = (hheise_placecell.SpatialInformation.ROI() & pk_pc).fetch('place_fields')

    # changed_trials = []
    # for idx, (a, n) in enumerate(zip(act, norm_act)):
    #     if a.shape[1] != n.shape[1]:
    #         changed_trials.append(idx)

    # Sort out sessions with less than 80 bins (in 170cm corridor)
    from itertools import compress
    mask = [True if x.shape[0] == 80 else False for x in norm_act_list]
    act_filt = list(compress(norm_act_list, mask))
    pk_filt = np.array(pk_pc)[mask]
    fields_filt = fields[mask]

    avg_act = np.vstack([np.mean(x, axis=1) for x in act_filt])

    # Sort out artefact neurons with maximum in last bin
    last_bin = [True if np.argmax(trace) != 79 else False for trace in avg_act]
    avg_act_filt = list(compress(avg_act, last_bin))
    pk_filt = pk_filt[last_bin]
    fields_filt = fields_filt[last_bin]

    # Sort neurons by activity maximum location
    sort_key = [(i, np.argmax(trace)) for i, trace in enumerate(avg_act_filt)]
    sort_key = [y[0] for y in sorted(sort_key, key=lambda x: x[1])]
    avg_act_filt_sort = np.array(avg_act_filt)[sort_key]

    # # Sort neurons by first place field bin (not pretty)
    # sort_key = [(i, field[0][0]) for i, field in enumerate(fields_filt)]
    # sort_key = [y[0] for y in sorted(sort_key, key=lambda x: x[1])]
    # avg_act_sort = avg_act[sort_key]

    # Normalize firing rate for each neuron (not pretty graph)
    # avg_act_sort_norm = avg_act_sort / np.amax(avg_act_sort, axis=1)[:,None]

    # Only keep neurons with a median FR lower than 33%, but high FR (90th percentile) higher than 80% of all neurons
    median_fr = np.median(avg_act_filt_sort, axis=1)
    median_33 = np.percentile(median_fr, 33)
    high_fr = np.percentile(avg_act_filt_sort, 90, axis=1)
    high_80 = np.percentile(high_fr, 80)

    sparse_neuron_mask = np.logical_and(median_fr < median_33, high_fr > high_80)
    print(np.sum(sparse_neuron_mask))

    # Plotting, formatting
    fig = plt.figure(figsize=(4.93, 7.3))
    # ax = sns.heatmap(gaussian_filter1d(avg_act_filt_sort[sparse_neuron_mask], sigma=1, axis=1), cmap='turbo',
    #                  vmax=15)  # Cap colormap a bit
    ax = sns.heatmap(avg_act_filt_sort[sparse_neuron_mask], cmap='turbo', vmin=-0.1, vmax=3)

    # Shade reward zones
    # zone_borders = (hheise_behav.CorridorPattern & 'pattern="training"').rescale_borders(n_bins=80)
    # for zone in zone_borders:
    #     ax.axvspan(zone[0], zone[1], facecolor='green', alpha=0.4)

    # Clean up axes and color bar
    ax.set_yticks([])
    ax.tick_params(axis=u'both', which=u'both', length=0)
    ax.set_xticks((0.5, 78.8), (0, 4), fontsize=20, rotation=0)
    ax.set_ylabel('Place cells', fontsize=20, labelpad=-3)
    ax.set_xlabel('Track position [m]', fontsize=20, labelpad=-20)

    cbar = ax.collections[0].colorbar
    cbar.ax.set_yticks((0, 2.9), (0, 3), fontsize=20)
    cbar.ax.tick_params(axis=u'both', which=u'both', length=0)
    # cbar.ax.set_ylabel('Firing rate [Hz]', fontsize=20, labelpad=-3, rotation=270)
    cbar.ax.set_ylabel(r'$\Delta$F/F', fontsize=20, labelpad=3, rotation=270)

    # Matrix has too many elements for Inkscape, use PNG of matrix instead of vectorized file
    plt.savefig(os.path.join(folder, 'figure1', 'all_place_cells_bartos_dff_8020_for_matrix.png'), transparent=True)
    # Much fewer cells, this file is used to load quickly into Inkscape, delete the matrix and use the axes
    plt.savefig(os.path.join(folder, 'figure1', 'all_place_cells_bartos_dff_3380_for_axes.svg'), transparent=True)


def plot_example_placecell():

    """
    Code to plot trials of a single session of a single example neuron. Not included in paper at the moment.
    ##########################
    ### Chosen place cell: ###
    ##########################
    # example_pc_keys = dict(username='hheise', mouse_id=115, day='2022-09-11', mask_id=292, place_cell_id=2)
    # example_pc_keys = dict(username='hheise', mouse_id=111, day='2022-08-21', mask_id=371, place_cell_id=2)
    example_pc_keys = dict(username='hheise', mouse_id=69, day='2021-02-28', mask_id=370, place_cell_id=2)

    ex_spikerate = (hheise_placecell.BinnedActivity.ROI() & example_pc_keys).fetch1('bin_spikerate')
    ex_act = (hheise_placecell.BinnedActivity.ROI() & example_pc_keys).fetch1('bin_activity')

    #######################################
    # Plot single-trial info as heatmap

    fig = plt.figure()
    ax = sns.heatmap(gaussian_filter1d(ex_spikerate.T, sigma=1, axis=1), cmap='jet')  # Cap colormap a bit
    # ax = sns.heatmap(avg_act_sort_norm, cmap='jet')

    # Shade reward zones
    zone_borders = (hheise_behav.CorridorPattern & 'pattern="training"').rescale_borders(n_bins=80)
    for zone in zone_borders:
        ax.axvspan(zone[0], zone[1], facecolor='green', alpha=0.4)

    # Clean up axes and color bar
    ax.set_yticks([])
    ax.tick_params(axis=u'both', which=u'both', length=0)
    ax.set_xticks((2, 78.8), (0, 4), fontsize=20, rotation=0)
    ax.set_yticks((0.9, ex_spikerate.shape[1] - 0.5), (1, ex_spikerate.shape[1]), fontsize=20, rotation=0)
    ax.set_ylabel('Trial no.', fontsize=20, labelpad=-25)
    ax.set_xlabel('Track position [m]', fontsize=20, labelpad=-20)

    cbar = ax.collections[0].colorbar
    cbar.ax.set_yticks((0.5, 18), (0, 18), fontsize=20)
    # cbar.ax.set_yticks((0, 2), (0, 2), fontsize=20)
    cbar.ax.tick_params(axis=u'both', which=u'both', length=0)
    cbar.ax.set_ylabel('Firing rate [Hz]', fontsize=20, labelpad=-5, rotation=270)
    # cbar.ax.set_ylabel(r'$\Delta$F/F', fontsize=20, labelpad=3, rotation=270)

    plt.savefig(os.path.join(folder, 'figure1',  'example_placecell_M69_20220228_370_fr_heatmap_1.svg'), transparent=True)

    ################################################
    # Plot single-trial spatial maps as lineplots
    stepsize = 1
    fig, ax = plt.subplots(1, 1)

    for i in range(1, ex_spikerate.shape[1] + 1):
        # ax.plot(gaussian_filter1d(ex_act[:, -i].T+i*stepsize, sigma=1), color='grey')
        ax.plot(ex_act[:, -i].T + i * stepsize, color='grey')

    # Shade reward zones
    zone_borders = (hheise_behav.CorridorPattern & 'pattern="training"').rescale_borders(n_bins=80)
    for zone in zone_borders:
        ax.axvspan(zone[0], zone[1], facecolor='green', alpha=0.4)

    # Clean up axes and color bar
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis=u'both', which=u'both', length=0)
    ax.set_xticks((0.5, 78.8), (0, 4), fontsize=20, rotation=0)
    ax.set_yticks((ex_spikerate.shape[1] + 0.8, 1.7), (1, ex_spikerate.shape[1]), fontsize=20, rotation=0)
    ax.set_ylabel('Trial no.', fontsize=20, labelpad=-25)
    ax.set_xlabel('Track position [m]', fontsize=20, labelpad=-20)

    plt.savefig(os.path.join(folder, 'figure1', 'example_placecell_M69_20220228_370_dff_lines.svg'), transparent=True)

    ##################
    # Plot raw traces
    dff = (common_img.Segmentation.ROI & example_pc_keys).fetch1('dff')
    running_masks, bin_frame_counts = (hheise_placecell.Synchronization.VRTrial & example_pc_keys).fetch('running_mask',
                                                                                                         'aligned_frames')
    n_bins, trial_mask = (hheise_placecell.PCAnalysis & example_pc_keys).fetch1('n_bins', 'trial_mask')

    # Split trace up into trials
    dff_trial = [dff[trial_mask == trial] for trial in np.unique(trial_mask)]

    # Get VR positions at frame times
    raw_pos = (hheise_behav.VRSession.VRTrial & example_pc_keys).get_arrays(['pos', 'frame'])
    pos = [tr[tr[:, 2] == 1, 1] + 10 for tr in raw_pos]

    n_trials = 8

    stepsize = 2
    fig, ax = plt.subplots(n_trials, 1, sharex='all')
    for i in range(1, n_trials + 1):
        ax[-i].plot(pos[-i][running_masks[-i]], color='blue')
        ax[-i].set_yticks([])
        ax[-i].spines['top'].set_visible(False)
        ax[-i].spines['left'].set_visible(False)
        ax[-i].spines['right'].set_visible(False)
        ax[-i].spines['bottom'].set_visible(False)
        ax2 = ax[-i].twinx()
        ax2.set_yticks([])
        ax2.plot(dff_trial[-i][running_masks[-i]], color='grey')
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        if i != 1:
            ax[-i].tick_params(axis=u'both', which=u'both', length=0)
            ax2.tick_params(axis=u'both', which=u'both', length=0)

    ax[0].set_xticks((0, 900), (0, int(900 / 30)), fontsize=20, rotation=0)
    ax2.vlines(800, ymin=1, ymax=2, label='1 dFF')
    plt.savefig(os.path.join(folder, 'figure1', 'example_placecell_M69_20220228_370_raw_dff_lines.svg'), transparent=True)

    """

    ########################
    # Heatmaps across sessions
    ########################

    # Load exported, cleaned data for Filippo
    with open(r'C:\Users\hheise.UZH\PycharmProjects\Caiman\custom scripts\preprint\Filippo\neural_data\spatial_activity_maps_dff.pkl', 'rb') as file:
        spat_act = pickle.load(file)
    with open(r'C:\Users\hheise.UZH\PycharmProjects\Caiman\custom scripts\preprint\Filippo\neural_data\is_pc.pkl', 'rb') as file:
        is_pc = pickle.load(file)
    with open(r'C:\Users\hheise.UZH\PycharmProjects\Caiman\custom scripts\preprint\Filippo\neural_data\pf_com.pkl', 'rb') as file:
        pf_com = pickle.load(file)

    # Merge dataframes for easier processing
    merge = []
    for key, df in pf_com.items():
        new_df = df.copy()
        new_df['mouse_id'] = key
        merge.append(new_df)
    merge_df = pd.concat(merge)

    mouse_ids = merge_df.pop('mouse_id').to_numpy()
    numeric_columns = merge_df.columns
    merge_df = merge_df.reindex(sorted(merge_df.columns), axis=1).reset_index()
    merge_df['mouse_id'] = mouse_ids

    # Define a function to replace empty lists or lists with more than 1 PF with NaN
    def replace_empty_list_with_nan(x):
        if isinstance(x, list) and len(x) != 1:
            return np.nan
        return x

    # Apply the function to the entire DataFrame
    merge_df = merge_df.applymap(replace_empty_list_with_nan)

    # Remove cells that are never a PC
    merge_df = merge_df.dropna(axis=0, how='all', subset=numeric_columns)

    # Go through each neuron, find subsequent days where the cell is a PC with one PF, and compute distances between PFs
    stable_pc_df = []
    for row_idx, row in merge_df.iterrows():
        local_ind = row.pop('index')
        mouse_id = row.pop('mouse_id')
        row_filt = row.dropna()
        if len(row_filt) > 1:
            pairs = list(combinations(row_filt.index, 2))
            for idx1, idx2 in pairs:
                stable_pc_df.append(pd.DataFrame([dict(mouse_id=mouse_id, local_idx=local_ind, day1=idx1, day2=idx2,
                                                       dist=np.abs(row_filt[idx1][0] - row_filt[idx2][0]))]))
    stable_pc_df = pd.concat(stable_pc_df, ignore_index=True)

    # Sort place cells based on distance
    stable_pc_sort = stable_pc_df.sort_values('dist')

    # Plot selected cell pairs
    def plot_cell_pairs(spatial_activity, pc_row):

        day1 = spatial_activity[pc_row.mouse_id].loc[pc_row.local_idx, pc_row.day1]
        day2 = spatial_activity[pc_row.mouse_id].loc[pc_row.local_idx, pc_row.day2]

        fig, ax = plt.subplots(nrows=2, ncols=1, sharex='all', sharey='all', layout='constrained')
        ax[0].plot(day1)
        ax[1].plot(day2)

        return day1, day2

    d1, d2 = plot_cell_pairs(spatial_activity=spat_act, pc_row=stable_pc_sort.loc[143])
    pd.Series(d2).to_clipboard(index=False, header=False)

    """
    Good indices:
    stable:
    2827, 1081
    remapping:
    1511, 170, 589
    """


def plot_performance_neural_correlations():
    # Select mice that should be part of the analysis (only mice that had successful microsphere surgery)

    # Get data from each mouse separately
    dfs = []
    for mouse in mouse_ids:

        # get date of microsphere injection
        surgery_day = (common_mice.Surgery() & 'surgery_type="Microsphere injection"' & f'mouse_id={mouse}').fetch1(
            'surgery_date')

        # Get the dates of 5 imaging sessions before the surgery, and all dates from 2 days after it
        pre_dates = (common_img.Segmentation & f'mouse_id={mouse}' & f'day < "{surgery_day.date()}"').fetch('day')[-5:]
        post_dates = np.unique(
            (common_img.Segmentation * hheise_behav.VRSession.VRTrial & 'pattern="training"' & f'mouse_id={mouse}'
             & f'day > "{surgery_day.date() + timedelta(days=1)}"').fetch('day'))
        dates = [*pre_dates, *post_dates]
        if len(dates) == 0:
            print(f'No segmentation for mouse {mouse}')
            continue
        is_pre = [*[True] * len(pre_dates), *[False] * len(post_dates)]

        # Get average performance for each session
        if mouse == 63:
            pre_dates_63 = (hheise_behav.VRPerformance & f'mouse_id={mouse}' & f'day < "{surgery_day.date()}"').fetch(
                'day')[
                           -5:]
            dates_63 = [*pre_dates_63, *post_dates]
            perf = (hheise_behav.VRPerformance & f'mouse_id={mouse}' & f'day in {helper.in_query(dates_63)}').get_mean()
        else:
            perf = (hheise_behav.VRPerformance & f'mouse_id={mouse}' & f'day in {helper.in_query(dates)}').get_mean()

        # Get mean and median firing rate
        curr_rate = [(common_img.ActivityStatistics.ROI & f'mouse_id={mouse}' & f'day="{day}"').fetch('rate_spikes')
                     for day in dates]
        is_empty = [True if len(x) == 0 else False for x in curr_rate]
        if any(is_empty):
            print(f'\nLength does not match for M{mouse} day {np.array(dates)[is_empty]}')
        mean_fr = [np.mean(x) for x in curr_rate]
        median_fr = [np.median(x) for x in curr_rate]

        # Get place cell ratio (Bartos and SI criteria)
        pc_bartos = (hheise_placecell.PlaceCell & f'mouse_id={mouse}' & 'corridor_type=0' & 'place_cell_id=2' &
                     f'day in {helper.in_query(dates)}').fetch('place_cell_ratio')
        pc_si = (hheise_placecell.SpatialInformation & f'mouse_id={mouse}' & 'corridor_type=0' & f'place_cell_id=2' &
                 f'day in {helper.in_query(dates)}').fetch('place_cell_ratio')

        mean_si = []
        median_si = []
        for date in dates:
            si = (
                        hheise_placecell.SpatialInformation.ROI & f'mouse_id={mouse}' & 'corridor_type=0' & f'place_cell_id=2' &
                        f'day = "{date}"').fetch('si')
            mean_si.append(np.mean(si))
            median_si.append(np.median(si))

        # Get within-session stability
        curr_rate = [(hheise_placecell.SpatialInformation.ROI & f'mouse_id={mouse}' & f'day="{day}"').fetch('stability')
                     for day in dates]
        mean_stab = [np.mean(x) for x in curr_rate]
        median_stab = [np.median(x) for x in curr_rate]
        sd_stab = [np.std(x) for x in curr_rate]

        # Build DataFrame
        data_dict = dict(mouse_id=[mouse] * len(dates), date=dates, is_pre=is_pre, performance=perf, mean_fr=mean_fr,
                         median_fr=median_fr, pc_si=pc_si, mean_stab=mean_stab, median_stab=median_stab,
                         sd_stab=sd_stab, pc_bartos=pc_bartos, mean_si=mean_si, median_si=median_si)
        try:
            data = pd.DataFrame(data_dict)
        except ValueError:
            if mouse == 63:
                data_dict = dict(mouse_id=[mouse] * len(perf), date=['2021-03-01'] * 5 + dates,
                                 is_pre=[True] * 5 + is_pre, performance=perf,
                                 mean_fr=[np.nan] * 5 + mean_fr,
                                 median_fr=[np.nan] * 5 + median_fr, pc_si=[np.nan] * 5 + list(pc_si),
                                 mean_stab=[np.nan] * 5 + mean_stab, median_stab=[np.nan] * 5 + median_stab,
                                 sd_stab=[np.nan] * 5 + sd_stab, pc_bartos=[np.nan] * 5 + list(pc_bartos),
                                 mean_si=[np.nan] * 5 + mean_si, median_si=[np.nan] * 5 + median_si)
                data = pd.DataFrame(data_dict)
            else:
                print(mouse)
                for key in data_dict.keys():
                    print(key, len(data_dict[key]))
                continue

        dfs.append(pd.DataFrame(data))

    data = pd.concat(dfs)

    data_group = data.merge(grouping)
    data_group = data_group.merge(grouping_2)

    # Plot correlations
    # for metric in ['mean_fr', 'median_fr', 'pc_bartos', 'pc_si']:
    for metric in ['mean_stab', 'pc_si', 'pc_bartos', 'mean_fr', 'mean_si', 'median_si']:
        sns.lmplot(data=data, x='performance', y=metric)
        rho, p = stats.spearmanr(data['performance'], data[metric])
        plt.text(0.12, 0.8, 'rho: {:.3f}\np: {:.5f}'.format(rho, p), transform=plt.gca().transAxes)

    # Plot data as group averages, pre and post
    metrics = ['performance', 'mean_stab', 'pc_si', 'pc_bartos']
    for metric in ['performance', 'mean_stab', 'pc_si', 'pc_bartos', 'mean_fr', 'mean_si', 'median_si']:
        plt.figure()
        sns.boxplot(data=data_group, x='group', y=metric, hue='is_pre', hue_order=[True, False],
                    order=['no_deficit', 'recovery', 'deficit_no_flicker', 'deficit_flicker'])
        rho, p = stats.spearmanr(data['performance'], data[metric])
        plt.text(0.12, 0.8, 'rho: {:.3f}\np: {:.5f}'.format(rho, p), transform=plt.gca().transAxes)

    # Compute mouse averages
    data_group_avg = []
    for mouse in data_group['mouse_id'].unique():
        for timepoint in [True, False]:
            new_df = dict(mouse_id=mouse, is_pre=timepoint)
            for metric in metrics:
                new_df[metric] = np.nanmean(
                    data_group[(data_group['mouse_id'] == mouse) & (data_group['is_pre'] == timepoint)][metric])
            data_group_avg.append(pd.DataFrame(new_df, index=[0]))
    data_group_avg = pd.concat(data_group_avg, ignore_index=True)
    data_group_avg = data_group_avg.merge(grouping_2)

    # Pivot for Prism export
    for metric in metrics:
        df_exp = data_group_avg.pivot(index='is_pre', columns='mouse_id', values=metric)
        if metric in ['pc_bartos', 'pc_si']:
            df_exp *= 100
        # Change order to align with groups
        df_exp_order = df_exp.loc[:, [91, 93, 94, 95, 33, 38, 83, 85, 86, 89, 90, 113, 41, 63, 69, 108, 110, 112]]
        df_exp_order = df_exp_order.reindex([True, False])
        df_exp_order.to_csv(os.path.join(folder, f'{metric}.csv'), sep='\t',
                            index=False)

    # Metrics that correlate well with performance: pc_si, stability (all 3)
    # Export for Prism
    data.to_csv(os.path.join(folder, 'figure1', 'neur_perf_corr.csv'), sep='\t',
                index=False,
                columns=['mouse_id', 'performance', 'pc_si', 'mean_stab'])


def plot_matched_cells_across_sessions_correlation_split(plot_sessions: List[int], groups: Tuple[str, str], compute_new: bool=False,
                                                         traces: Optional[list]=None, is_pc_list: Optional[list]=None,
                                                         match_matrices: Optional[list]=None, sort_session: int = 0,
                                                         quantiles: Tuple[float, float] = (0.25, 0.75),
                                                         rel_noncoding_size: int = 2, cross_validation: bool=False,
                                                         normalize: Optional[str] = None, titles: Optional[Iterable] = None,
                                                         smooth: Optional[int] = None, gaussian_sigma: Optional[int] = None, cmap='turbo') -> plt.Figure:
    """
    Plot traces of matched neurons across sessions. Neurons are sorted by location of max activity in a given session.

    Args:
        plot_sessions: List of days after injection (rel_days) to plot
        groups: List with 2 elements, names of behavioral groups to plot in the upper and lower row
        compute_new: Bool flag whether data should be computed new or loaded from file.
        traces: Traces to plot, loaded from data cleaner pickle.
        is_pc_list: Which cells are place cells, loaded from data cleaner pickle.
        match_matrices: Matched Index Dataframe, loaded from data cleaner pickle.
        sort_session: Day after injection where sorting should take place.
        quantiles: Tuple of 2 values giving lower and upper quantile of correlation for stable/unstable split
        rel_noncoding_size: Number of noncoding cells to plot, relative to number of place cells plotted
        cross_validation:
        normalize: What kind of activity normalization should be performed. Either None, 'neuron', or 'neuron_session'.
        titles: List of titles for each subplot/session.
        smooth: Bool Flag whether the activity should be smoothed, and with which sigma.
        cmap: Color map used to plot the traces.

    Returns:
        Matplotlib figure
    """

    def plot_heatmap(data_to_plot, axis, group_idx, day_idx, group_name):
        sns.heatmap(data_to_plot, ax=axis, cbar=False, vmin=vmin, vmax=vmax, cmap=cmap)

        for z in zones:
            axis.axvline(z[0], linestyle='--', c='green')
            axis.axvline(z[1], linestyle='--', c='green')

        if day_idx != 0:
            axis.set_yticks([])
        else:
            axis.set_ylabel(group_name, fontsize=15, labelpad=-3)

        axis.set_xticks([])
        if group_idx == len(groups) - 1:
            axis.set_xlabel('Track position [m]', fontsize=15)
            # ax.set_title(f'{traces_sort.shape[1]-idx}d prestroke')

        if (group_idx == 0) and (titles is not None):
            axis.set_title(titles[day_idx])

    # Build arrays
    sort_session_idx = np.where(np.array(plot_sessions) == sort_session)[0][0]

    if compute_new:
        dfs = []
        for mouse_dict, is_pc_dict, match_dict in zip(traces, is_pc_list, match_matrices):

            # Extract data from dicts, align session days
            mouse_id = list(mouse_dict.keys())[0]
            if mouse_id == '121_1':
                continue
            rel_days = np.array(mouse_dict[mouse_id][1])
            trace = mouse_dict[mouse_id][0]
            is_pc_m = is_pc_dict[mouse_id][0]

            if 3 not in rel_days:
                rel_days[(rel_days == 2) | (rel_days == 4)] = 3
            rel_days[(rel_days == 5) | (rel_days == 6) | (rel_days == 7)] = 6
            rel_days[(rel_days == 8) | (rel_days == 9) | (rel_days == 10)] = 9
            rel_days[(rel_days == 11) | (rel_days == 12) | (rel_days == 13)] = 12
            rel_days[(rel_days == 14) | (rel_days == 15) | (rel_days == 16)] = 15
            rel_days[(rel_days == 17) | (rel_days == 18) | (rel_days == 19)] = 18
            rel_days[(rel_days == 20) | (rel_days == 21) | (rel_days == 22)] = 21
            rel_days[(rel_days == 23) | (rel_days == 24) | (rel_days == 25)] = 24
            if 28 not in rel_days:
                rel_days[(rel_days == 26) | (rel_days == 27) | (rel_days == 28)] = 27

            rel_sess = np.arange(len(rel_days)) - np.argmax(np.where(rel_days <= 0, rel_days, -np.inf))
            rel_days[(np.min(rel_sess) <= rel_sess) & (rel_sess < 1)] = np.arange(np.min(rel_sess), 1)
            first_post_idx = np.where(np.array(plot_sessions) > 0)[0][0]

            # Make mask to only use cells that appear in all sessions to be plotted
            trace_plot = trace[:, np.isin(rel_days, plot_sessions)]
            is_pc_plot = is_pc_m[:, np.isin(rel_days, plot_sessions)]
            cell_mask = np.sum(np.isnan(trace_plot[:, :, 0]), axis=1) == 0
            trace_plot = trace_plot[cell_mask]
            is_pc_plot = is_pc_plot[cell_mask]
            sort_sess_mask_idx = match_dict[mouse_id].iloc[cell_mask, np.isin(rel_days, plot_sessions)].iloc[:, sort_session_idx]

            if trace_plot.shape[1] != len(plot_sessions):
                continue

            # Fetch spatial activity maps from odd and even trials of sorting session separately
            sort_sess_key = common_match.MatchedIndex().string2key(sort_sess_mask_idx.name)

            restriction = [dict(mouse_id=int(mouse_id.split("_")[0]), **sort_sess_key, mask_id=i, place_cell_id=2) for i in sort_sess_mask_idx]
            db_mask_id, spat_sorting = (hheise_placecell.BinnedActivity.ROI & restriction).fetch('mask_id', 'bin_activity')

            # Create a mapping from values to their indices in the unsorted array
            value_to_index = {value: index for index, value in enumerate(sort_sess_mask_idx)}

            # Use the mapping to find indices in the unsorted array for elements in the sorted array
            indices_in_unsorted_array = np.array([value_to_index[value] for value in db_mask_id])

            # Use the index mapping to reorder spat_sorting to match the order in trace_plot
            spat_sorting_sorted = np.stack(spat_sorting[indices_in_unsorted_array])

            # Compute spatial activity map from odd and even trials
            spat_sorting_even = np.mean(spat_sorting_sorted[:, :, ::2], axis=2)
            spat_sorting_odd = np.mean(spat_sorting_sorted[:, :, 1::2], axis=2)

            # Substitute the full spatial map in the sorting session with only odd trial for cross-validation
            # trace_plot[:, sort_session_idx, :] = spat_sorting_odd

            # For each neuron, correlate all session pairs
            for neur_idx, cell in enumerate(trace_plot):
                if smooth is not None:
                    cor_mat = np.corrcoef(ndimage.gaussian_filter1d(cell, sigma=smooth, axis=1))
                else:
                    cor_mat = np.corrcoef(cell)
                cor_mat[np.triu_indices(cor_mat.shape[0], 0)] = np.nan
                dfs.append(pd.DataFrame([dict(mouse_id=int(mouse_id.split("_")[0]), session_str=sort_sess_mask_idx.name,
                                              mask_id=int(sort_sess_mask_idx.iloc[neur_idx]),
                                              is_pc=int(is_pc_plot[neur_idx, sort_session_idx]),
                                              pre_cor=np.tanh(np.nanmean(np.arctanh(cor_mat[:first_post_idx]))),
                                              post_cor=np.tanh(np.nanmean(np.arctanh(cor_mat[first_post_idx:]))),
                                              all_cor=np.tanh(np.nanmean(np.arctanh(cor_mat))), traces=cell,
                                              spat_sorting_even=spat_sorting_even[neur_idx],
                                              spat_sorting_odd=spat_sorting_odd[neur_idx])]))
        df = pd.concat(dfs, ignore_index=True)

        df.to_pickle(r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Papers\preprint\class_quantification\example_plot_data_smooth.pickle')

    elif smooth is not None:
        df = pd.read_pickle(r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Papers\preprint\class_quantification\example_plot_data_smooth.pickle')
    elif cross_validation:
        df = pd.read_pickle(r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Papers\preprint\class_quantification\example_plot_data_crossval.pickle')
    else:
        df = pd.read_pickle(r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Papers\preprint\class_quantification\example_plot_data.pickle')

    # Enter groups
    coarse = (hheise_grouping.BehaviorGrouping & 'grouping_id = 4' & 'cluster = "coarse"').get_groups()
    fine = (hheise_grouping.BehaviorGrouping & 'grouping_id = 4' & 'cluster = "fine"').get_groups()
    df = df.merge(coarse, how='left', on='mouse_id').rename(columns={'group': 'coarse'})
    df = df.merge(fine, how='left', on='mouse_id').rename(columns={'group': 'fine'})

    zones = (hheise_behav.CorridorPattern() & 'pattern="training"').rescale_borders(80)

    # Create plot
    fig, ax = plt.subplots(nrows=len(groups), ncols=len(plot_sessions), sharex='all', sharey='row', figsize=(20, 13),
                           layout='constrained')

    # For each group, sort and split cells, and plot traces
    for j, group in enumerate(groups):

        if group in coarse.group.values:
            df_filt = df[df.coarse == group]
        elif group in fine.group.values:
            df_filt = df[df.fine == group]
        else:
            raise IndexError(f'Group {group} not found.')

        # if cross_validation:
        #     all_traces = list(df_filt.loc[:, "traces"].copy())
        #     all_even = list(df_filt.loc[:, "spat_sorting_even"].copy())
        #     all_new_traces = []
        #     for all_trace, all_eve in zip(all_traces, all_even):
        #         all_trace = all_trace.copy()
        #         all_trace[sort_session_idx] = all_eve
        #         all_new_traces.append(all_trace)
        #     df_filt['traces'] = all_new_traces

        # Get 25th and 75th quantile of place cell correlation
        lower_quant = np.quantile(df_filt[df_filt.is_pc == 1]['all_cor'], quantiles[0])
        upper_quant = np.quantile(df_filt[df_filt.is_pc == 1]['all_cor'], quantiles[1])

        stable_df = df_filt[(df_filt.is_pc == 1) & (df_filt.all_cor > upper_quant)]
        unstable_df = df_filt[(df_filt.is_pc == 1) & (df_filt.all_cor < lower_quant)]

        # Randomly select subset of noncoding cells
        idx = np.random.choice(np.arange((df_filt.is_pc == 0).sum()), len(stable_df['traces'])*2*rel_noncoding_size, replace=False)
        noncoding_traces = np.stack(df_filt[(df_filt.is_pc == 0)]['traces'].values)[idx]

        # Sort traces based on sort_session
        def sort_traces(key_trace, sort_trace):
            sort_key = [(i, np.argmax(trace)) for i, trace in enumerate(key_trace)]
            sort_key = [y[0] for y in sorted(sort_key, key=lambda x: x[1])]
            return sort_trace[sort_key]

        # Sort via cross-validation
        if cross_validation:
            stable_traces_sort = sort_traces(np.stack(stable_df['spat_sorting_even'].values),
                                             np.stack(stable_df['traces'].values))
            unstable_traces_sort = sort_traces(np.stack(unstable_df['spat_sorting_even'].values),
                                               np.stack(unstable_df['traces'].values))
            noncoding_traces_sort = sort_traces(np.stack(df_filt[(df_filt.is_pc == 0)]['spat_sorting_even'].values)[idx],
                                                noncoding_traces)
        else:
            # Sort without cross-validation
            stable_traces_sort = sort_traces(np.stack(stable_df['traces'].values)[:, sort_session_idx],
                                             np.stack(stable_df['traces'].values))
            unstable_traces_sort = sort_traces(np.stack(unstable_df['traces'].values)[:, sort_session_idx],
                                               np.stack(unstable_df['traces'].values))
            noncoding_traces_sort = sort_traces(np.stack(df_filt[(df_filt.is_pc == 0)]['traces'].values)[idx][:, sort_session_idx],
                                                noncoding_traces)

        # Combine arrays with an empty row
        data_arrays = [stable_traces_sort, unstable_traces_sort, noncoding_traces_sort]
        empty_row = np.zeros((1, *data_arrays[0].shape[1:])) * np.nan
        stacked_data = [np.vstack([x, empty_row]) for x in data_arrays[:-1]]
        stacked_data.append(data_arrays[-1])
        stacked_data = np.vstack(stacked_data)

        # Normalize whole plot
        if normalize == 'all':
            vmin = np.nanmin(stacked_data)
            vmax = np.nanmax(stacked_data)

        # Normalize activity neuron-wise for each session separately
        elif normalize == 'neuron':
            traces_norm = []
            for i in range(stacked_data.shape[1]):
                neur_sess_max = np.nanmax(stacked_data[:, i, :], axis=1)
                neur_sess_min = np.nanmin(stacked_data[:, i, :], axis=1)
                traces_norm.append((stacked_data[:, i, :] - neur_sess_min[:, None]) /
                                   (neur_sess_max[:, None] - neur_sess_min[:, None]))
            stacked_data = np.stack(traces_norm, axis=1)
            vmin = 0
            vmax = 1

        # Normalize activity neuron-wise across sessions
        elif normalize == 'neuron_session':
            sess_squeeze = np.reshape(stacked_data, (stacked_data.shape[0], stacked_data.shape[1] * stacked_data.shape[2]))
            neur_max = np.nanmax(sess_squeeze, axis=1)
            neur_min = np.nanmin(sess_squeeze, axis=1)
            stacked_data = (stacked_data - neur_min[:, None, None]) / (neur_max[:, None, None] - neur_min[:, None, None])
            vmin = 0
            vmax = 1

        else:
            vmin = None
            vmax = None

        # Plot traces for each day as heatmaps
        for i in range(stacked_data.shape[1]):

            if gaussian_sigma is not None:
                plot_data = ndimage.gaussian_filter1d(stacked_data[:, i], axis=1, sigma=gaussian_sigma)
            else:
                plot_data = stacked_data[:, i]

            if normalize == 'session':
                plot_data = (plot_data-np.nanmin(plot_data))/(np.nanmax(plot_data)-np.nanmin(plot_data))
                vmin = 0
                vmax = 1

            # sns.heatmap(stacked_data[:, i], ax=ax[j, i], cbar=False, cmap='turbo')
            if len(groups) == 1:
                plot_heatmap(plot_data, axis=ax[i], group_idx=j, day_idx=i, group_name=group)
            else:
                plot_heatmap(plot_data, axis=ax[j, i], group_idx=j, day_idx=i, group_name=group)

    return fig


def find_example_cells_for_classes():

    def create_plot(traces, row_ids, n_sess=3, offset=0.5, xrange=(0, 80), figsize=(39, 140)):
        mm = 1 / 2.54 / 10  # millimeters in inches
        fig, ax = plt.subplots(nrows=len(row_ids), ncols=1, sharey='all', sharex='all',
                               figsize=(figsize[0] * mm, figsize[1] * mm), layout='constrained')
        zones = (hheise_behav.CorridorPattern() & 'pattern="training"').rescale_borders(80)


        for i, cell in enumerate(row_ids):
            cell_trace = traces.loc[cell][:n_sess]
            for sess in range(len(cell_trace)):
                ax[i].plot(cell_trace[sess] - offset*sess)
            ax[i].set_title(cell)
            ax[i].set_xlim(xrange)

            for z in zones:
                ax[i].axvline(z[0], linestyle='--', c='green')
                ax[i].axvline(z[1], linestyle='--', c='green')

        return fig



    df = pd.read_pickle(r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Papers\preprint\class_quantification\example_plot_data.pickle')

    # Enter groups
    coarse = (hheise_grouping.BehaviorGrouping & 'grouping_id = 4' & 'cluster = "coarse"').get_groups()
    fine = (hheise_grouping.BehaviorGrouping & 'grouping_id = 4' & 'cluster = "fine"').get_groups()
    df = df.merge(coarse, how='left', on='mouse_id').rename(columns={'group': 'coarse'})
    df = df.merge(fine, how='left', on='mouse_id').rename(columns={'group': 'fine'})

    zones = (hheise_behav.CorridorPattern() & 'pattern="training"').rescale_borders(80)

    # Get stable and unstable cells from most/least correlated pre-stroke cells
    df_sort = df[(df.is_pc == 1) & (df.mouse_id == 116)].sort_values(by='pre_cor', ascending=False)

    # Plot traces to find good example
    idx = -17
    plt.figure(); sns.heatmap(df_sort.traces.iloc[idx]).set_title(df_sort.index[idx])

    idx = 5
    plt.figure(); sns.heatmap(df[(df.is_pc == 0) & (df.mouse_id == 116)].traces.iloc[idx]).set_title(df[(df.is_pc == 0) & (df.mouse_id == 116)].index[idx])

    """
    Row idx of df:
    M41:
        stable cell: 163 (M41, 2020-08-24, mask_id 358)
        unstable cell: 207 (M41, 2020-08-24, mask_id 227)
        noncoding cell: 160 (M41, 2020-08-24, mask_id 345)
    M116: --> used in figure 1H
        stable cell: 1176 (M116, 2022-08-09, mask_id 1405)
        unstable cell: 1208 (M116, 2022-08-09, mask_id 757)
        noncoding cell: 1057 (M116, 2022-08-09, mask_id 246)
    """

    # Plot candidate cells together to see if they make a nice plot
    fig = create_plot(df.traces, [1176, 1208, 1057], n_sess=4, offset=1, xrange=(13, 45), figsize=(35, 140))
    fig.savefig(r"C:\Users\hheise.UZH\Desktop\preprint\figure1\place_cell_example_traces.svg")


    # Extract activity data from first four sessions and save it for prism
    pd.DataFrame(df.traces.loc[163][:4]).T.to_clipboard(index=False, header=False)
    pd.DataFrame(df.traces.loc[207][:4]).T.to_clipboard(index=False, header=False)
    pd.DataFrame(df.traces.loc[170][:4]).T.to_clipboard(index=False, header=False)


# Plot matched cells across days, split into groups
plot_matched_cells_across_sessions_correlation_split(plot_sessions=[-2, -1, 0, 3, 6, 12, 15], groups=("Stroke",),
                                                     normalize='all', gaussian_sigma=1, titles=[-2, -1, 0, 3, 6, 12, 15])
plt.savefig(os.path.join(folder, 'figure1\\stable_unstable_crosssession_smooth_nocrossval_session_normalized_onlyStroke.png'))

# Changed function call to find examples of remapping and stable PCs
plot_matched_cells_across_sessions_correlation_split(plot_sessions=[-3, -2, -1, 0], groups=("Sham",),
                                                     normalize='all', gaussian_sigma=1, titles=[-3, -2, -1, 0])


def get_number_pc_per_session():
    with open(r'C:\Users\hheise.UZH\PycharmProjects\Caiman\custom scripts\preprint\Filippo\neural_data\is_pc_all_cells.pkl', 'rb') as file:
        is_pc_all_cells = pickle.load(file)

    # Count place cells for each mouse
    pc_counts = {}
    for mouse_id, mouse_df in is_pc_all_cells.items():
        pc_counts[mouse_id] = mouse_df.sum(axis=0).rename(mouse_id)

    pc_count_df = pd.concat(pc_counts).reset_index().pivot(index='level_0', columns='level_1', values=0)
