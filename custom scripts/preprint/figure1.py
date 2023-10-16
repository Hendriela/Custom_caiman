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
from typing import Optional, Tuple
import os
from datetime import timedelta
from scipy.ndimage import gaussian_filter1d
import tifffile as tif
import standard_pipeline.performance_check as performance
from itertools import combinations
import pandas as pd
import seaborn as sns
from scipy import stats
import numpy as np

from schema import hheise_behav, common_mice, hheise_placecell, common_img, common_match, hheise_hist
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



def example_lick_histogram():
    ### Example lick histogram

    # Set session paths
    paths = [r"G:\Batch5\M63\20210214",
             r"G:\Batch5\M63\20210302",
             r"G:\Batch5\M63\20210306",
             r"G:\Batch5\M63\20210317"]

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

    plt.savefig(os.path.join(folder, 'example_VR_behavior.svg'), transparent=True)


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
    pk_pc = (common_img.Segmentation.ROI * hheise_placecell.PlaceCell.ROI * hheise_placecell.SpatialInformation.ROI & restrictions &
             'snr>10' & 'r>0.9' & 'cnn>0.9').fetch('KEY')

    # Get spatial activity maps
    act = (hheise_placecell.BinnedActivity.ROI() & pk_pc).fetch('bin_spikerate')
    # Only keep traces from normal corridor trials
    norm_act = np.array([curr_act[:, (hheise_behav.VRSession & cell_pk).get_normal_trials()]
                         for curr_act, cell_pk in zip(act, pk_pc)], dtype='object')
    fields = (hheise_placecell.SpatialInformation.ROI() & pk_pc).fetch('place_fields')

    # changed_trials = []
    # for idx, (a, n) in enumerate(zip(act, norm_act)):
    #     if a.shape[1] != n.shape[1]:
    #         changed_trials.append(idx)

    # Sort out sessions with less than 80 bins (in 170cm corridor)
    mask = [True if x.shape[0] == 80 else False for x in norm_act]
    act_filt = act[mask]
    pk_filt = np.array(pk_pc)[mask]
    fields_filt = fields[mask]

    avg_act = np.vstack([np.mean(x, axis=1) for x in act_filt])

    # Sort out artefact neurons with maximum in last bin
    last_bin = [True if np.argmax(trace) != 79 else False for trace in avg_act]
    avg_act_filt = avg_act[last_bin]
    pk_filt = pk_filt[last_bin]
    fields_filt = fields_filt[last_bin]

    # Sort neurons by activity maximum location
    sort_key = [(i, np.argmax(trace)) for i, trace in enumerate(avg_act_filt)]
    sort_key = [y[0] for y in sorted(sort_key, key=lambda x: x[1])]
    avg_act_filt_sort = avg_act_filt[sort_key]

    # # Sort neurons by first place field bin (not pretty)
    # sort_key = [(i, field[0][0]) for i, field in enumerate(fields_filt)]
    # sort_key = [y[0] for y in sorted(sort_key, key=lambda x: x[1])]
    # avg_act_sort = avg_act[sort_key]

    # Normalize firing rate for each neuron (not pretty graph)
    # avg_act_sort_norm = avg_act_sort / np.amax(avg_act_sort, axis=1)[:,None]

    # Only keep neurons with a median FR lower than 33%, but high FR (90th percentile) higher than 80% of all neurons
    median_fr = np.median(avg_act_filt_sort, axis=1)
    median_33 = np.percentile(median_fr, 80)
    high_fr = np.percentile(avg_act_filt_sort, 90, axis=1)
    high_80 = np.percentile(high_fr, 20)

    sparse_neuron_mask = np.logical_and(median_fr < median_33, high_fr > high_80)
    print(np.sum(sparse_neuron_mask))

    # Plotting, formatting
    fig = plt.figure(figsize=(4.93, 7.3))
    ax = sns.heatmap(gaussian_filter1d(avg_act_filt_sort[sparse_neuron_mask], sigma=1, axis=1), cmap='turbo',
                     vmax=15)  # Cap colormap a bit
    # ax = sns.heatmap(avg_act_sort_norm, cmap='jet')

    # Shade reward zones
    # zone_borders = (hheise_behav.CorridorPattern & 'pattern="training"').rescale_borders(n_bins=80)
    # for zone in zone_borders:
    #     ax.axvspan(zone[0], zone[1], facecolor='green', alpha=0.4)

    # Clean up axes and color bar
    ax.set_yticks([])
    ax.tick_params(axis=u'both', which=u'both', length=0)
    ax.set_xticks((0.5, 78.8), (0, 4), fontsize=20, rotation=0)
    ax.set_ylabel('Cell no.', fontsize=20, labelpad=-3)
    ax.set_xlabel('Track position [m]', fontsize=20, labelpad=-20)

    cbar = ax.collections[0].colorbar
    cbar.ax.set_yticks((0.5, 14.5), (0, 15), fontsize=20)
    cbar.ax.tick_params(axis=u'both', which=u'both', length=0)
    cbar.ax.set_ylabel('Firing rate [Hz]', fontsize=20, labelpad=-3, rotation=270)

    # Matrix has too many elements for Inkscape, use PNG of matrix instead of vectorized file
    plt.savefig(os.path.join(folder, 'figure1', 'all_place_cells_bartos_8020_for_matrix.png'), transparent=True)
    # Much fewer cells, this file is used to load quickly into Inkscape, delete the matrix and use the axes
    plt.savefig(os.path.join(folder, 'figure1', 'all_place_cells_3380_for_axes.svg'), transparent=True)


def plot_example_placecell():

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


