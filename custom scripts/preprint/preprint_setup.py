#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 15/12/2022 12:07
@author: hheise

Plots for the preprint December 2022
"""
import matplotlib
# matplotlib.use('TkAgg')
matplotlib.rcParams['font.sans-serif'] = "Arial"    # Use same font as Prism
matplotlib.rcParams['font.family'] = "sans-serif"

from matplotlib import pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from typing import Optional
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
from schema import hheise_behav, common_mice, hheise_placecell, common_img, common_match
from util import helper

mouse_ids = [33, 38, 41,    # Batch 3
             63, 69,        # Batch 5
             83, 85, 86, 89, 90, 91, 93, 94, 95,  # Batch 7
             108, 110, 111, 112, 113, 114, 115, 116, 120, 122]  # Batch 8

no_deficit = [93, 91, 94, 95, 109, 123, 120]
no_deficit_flicker = [111, 114, 116]
recovery = [33, 38, 83, 85, 86, 89, 90, 113]
deficit_no_flicker = [41, 63, 69, 121]
deficit_flicker = [108, 110, 112]
sham_injection = [115, 122]

mice = [*no_deficit, *no_deficit_flicker, *recovery, *deficit_no_flicker, *deficit_flicker, *sham_injection]

grouping = pd.DataFrame(data=dict(mouse_id=mice,
                                  group=[*['no_deficit']*len(no_deficit), *['no_deficit_flicker']*len(no_deficit_flicker),
                                         *['recovery']*len(recovery), *['deficit_no_flicker']*len(deficit_no_flicker),
                                         *['deficit_flicker']*len(deficit_flicker), *['sham_injection']*len(sham_injection)]))
grouping_2 = pd.DataFrame(data=dict(mouse_id=mice,
                                      group=[*['no_deficit']*len(no_deficit), *['no_deficit']*len(no_deficit_flicker),
                                             *['recovery']*len(recovery), *['deficit_no_flicker']*len(deficit_no_flicker),
                                             *['deficit_flicker']*len(deficit_flicker), *['no_deficit']*len(sham_injection)]))

folder = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Papers\preprint'


#%% Figure 1
def example_fov():
    avg = (common_img.QualityControl & 'mouse_id=41' & 'day="2020-08-24"').fetch1('avg_image')
    tif.imwrite(os.path.join(folder, '41_20200824_fov_overview.tif'), avg)

    avg = (common_img.QualityControl & 'mouse_id=41' & 'day="2020-08-26"').fetch1('avg_image')
    tif.imwrite(os.path.join(folder, '41_20200824_fov_overview_damage.tif'), avg)


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
        days, perf = (hheise_behav.VRPerformance & 'username="hheise"' &
                      f'mouse_id={mouse}' & f'day<"{surgery_day}"').fetch('day', 'binned_lick_ratio')
        # Calculate average performance for each day
        mean_perf = [np.mean(x) for x in perf]
        # Transform dates into days before surgery
        rel_days = [(d - surgery_day).days for d in days]
        # Get session number before surgery
        rel_sess = np.arange(-len(mean_perf), 0)

        dfs.append(pd.DataFrame(dict(mouse_id=mouse, days=days, rel_days=rel_days, rel_sess=rel_sess, perf=mean_perf)))
    df = pd.concat(dfs, ignore_index=True)

    # sns.lineplot(data=df_all, x='rel_sess', y='perf')

    #Export for prism
    df_exp = df.pivot(index='rel_sess', columns='mouse_id', values='perf')
    df_exp.to_csv(os.path.join(folder, 'learning_curve.csv'), sep='\t')


def place_cell_plot():

    # Get primary keys of all place cells in the normal corridor, apply strict quality criteria
    thresh = (common_img.Segmentation.ROI * hheise_placecell.SpatialInformation.ROI & 'is_pc=1' & 'accepted=1').fetch('pf_threshold')
    thresh_quant = np.quantile(thresh, 0.95)
    restrictions = dict(is_pc=1, accepted=1, p_si=0, p_stability=0, corridor_type=0)
    pk_pc_si = (common_img.Segmentation.ROI * hheise_placecell.SpatialInformation.ROI & restrictions &
             'snr>10' & 'r>0.9' & 'cnn>0.9' & f'pf_threshold>{thresh_quant}').fetch('KEY')

    # Get place cells from Bartos criteria
    restrictions = dict(is_place_cell=1, accepted=1, corridor_type=0, place_cell_id=2)
    pk_pc = (common_img.Segmentation.ROI * hheise_placecell.PlaceCell.ROI & restrictions &
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
    ax = sns.heatmap(gaussian_filter1d(avg_act_filt_sort[sparse_neuron_mask], sigma=1, axis=1), cmap='jet', vmax=15)   # Cap colormap a bit
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

    plt.savefig(os.path.join(folder, 'all_place_cells_bartos_8020.png'), transparent=True)
    plt.savefig(os.path.join(folder, 'all_place_cells_3380.svg'), transparent=True)


def plot_example_placecell():

    def screen_place_cells():
        # Spatial info based
        # thresh = (common_img.Segmentation.ROI * hheise_placecell.SpatialInformation.ROI &
        #           'is_pc=1' & 'accepted=1').fetch('pf_threshold')
        # thresh_quant = np.quantile(thresh, 0.95)
        # restrictions = dict(is_pc=1, accepted=1, p_si=0, p_stability=0, corridor_type=0)
        # pk_pc = (common_img.Segmentation.ROI * hheise_placecell.SpatialInformation.ROI &
        #          restrictions & f'pf_threshold>{thresh_quant}').fetch('KEY')

        # Bartos criteria
        restrictions = dict(is_place_cell=1, accepted=1, corridor_type=0, place_cell_id=2, p=0)
        pk_pc = (common_img.Segmentation.ROI * hheise_placecell.PlaceCell.ROI & restrictions &
                 'snr>10' & 'r>0.9' & 'cnn>0.9').fetch('KEY')

        ### Find good placecell from a session with many trials
        # Get sessions with >= 20 trials
        norm_trials, pk = (hheise_behav.VRSession & pk_pc).get_normal_trials(include_pk=True)
        long_sess = [pk[i] for i in range(len(norm_trials)) if len(norm_trials[i])>=10]

        # Get good place cells only from these sessions
        good_pk_pc = np.array((common_img.Segmentation.ROI * hheise_placecell.SpatialInformation.ROI &
                      restrictions & long_sess).fetch('KEY'))

        good_pk_pc = np.array((common_img.Segmentation.ROI * hheise_placecell.PlaceCell.ROI &
                      restrictions & long_sess & 'snr>10' & 'r>0.9' & 'cnn>0.9').fetch('KEY'))

        # Get spatial activity maps
        act = (hheise_placecell.BinnedActivity.ROI() & good_pk_pc).fetch('bin_spikerate')
        # Filter out non-standard trials
        norm_act = np.array([curr_act[:, (hheise_behav.VRSession & cell_pk).get_normal_trials()]
                             for curr_act, cell_pk in zip(act, good_pk_pc)], dtype='object')
        fields = (hheise_placecell.SpatialInformation.ROI() & good_pk_pc).fetch('place_fields')

        # Sort out sessions with less than 20 trials
        mask = [True if x.shape[1] >= 10 else False for x in norm_act]
        act_filt = norm_act[mask]
        # act_filt = [b for a, b in zip(mask, norm_act) if a]
        good_pk_pc_filt = good_pk_pc[mask]
        # fields_filt = fields[mask]

        # Sort out sessions with less than 80 bins (in 170cm corridor)
        mask = [True if x.shape[0] == 80 else False for x in act_filt]
        act_filt = act_filt[mask]
        good_pk_pc_filt = good_pk_pc_filt[mask]
        # fields_filt = fields_filt[mask]

        # Sort out artefact neurons with maximum in last bin
        avg_act = np.vstack([np.mean(x, axis=1) for x in act_filt])
        last_bin = [True if np.argmax(trace) != 79 else False for trace in avg_act]
        act_filt = act_filt[last_bin]
        good_pk_pc_filt = good_pk_pc_filt[last_bin]
        avg_act = avg_act[last_bin]

        # Only keep neurons with a median FR lower than 33%, but high FR (90th percentile) higher than 80% of all neurons
        median_fr = np.median(avg_act, axis=1)
        median_33 = np.percentile(median_fr, 33)
        high_fr = np.percentile(avg_act, 90, axis=1)
        high_80 = np.percentile(high_fr, 80)
        sparse_neuron_mask = np.logical_and(median_fr < median_33, high_fr > high_80)

        act_filt = act_filt[sparse_neuron_mask]
        good_pk_pc_filt = good_pk_pc_filt[sparse_neuron_mask]

        # Sort out artefact neurons with maximum in last bin
        # last_bin = [True if np.argmax(trace) != 79 else False for trace in avg_act]
        # act_filt = [b for a, b in zip(last_bin, act_filt) if a]
        # good_pk_pc_filt = good_pk_pc_filt[last_bin]
        # fields_filt = fields_filt[last_bin]

        # Sort neurons by difference between maximum and minimum firing rate
        # sorting = [(y, x, z) for y, x, z in sorted(zip(act_filt, good_pk_pc_filt, fields_filt), key=lambda pair: np.quantile(pair[0], 0.99)-np.mean(pair[0]), reverse=True)]
        # sorting = [(y, x, z) for y, x, z in sorted(zip(act_filt, good_pk_pc_filt, fields_filt),
        #                                            key=lambda pair: np.quantile(pair[0], 0.8))]
        sorting = [(y, x) for y, x in sorted(zip(act_filt, good_pk_pc_filt),
                                                   key=lambda pair: np.quantile(pair[0], 0.8))]
        # act_sort, pk_sort, fields_sort = zip(*sorting)
        act_sort, pk_sort = zip(*sorting)

        # Plot sample neuron
        idx = 8

        # Plotting, formatting
        fig = plt.figure()
        ax = sns.heatmap(gaussian_filter1d(act_sort[idx].T, sigma=1, axis=1), cmap='jet')     # Cap colormap a bit
        # ax = sns.heatmap(avg_act_sort_norm, cmap='jet')

        # Shade reward zones
        zone_borders = (hheise_behav.CorridorPattern & 'pattern="training"').rescale_borders(n_bins=80)
        for zone in zone_borders:
            ax.axvspan(zone[0], zone[1], facecolor='green', alpha=0.4)

        # Clean up axes and color bar
        ax.set_title(f'M{pk_sort[idx]["mouse_id"]}_{pk_sort[idx]["day"]}_mask{pk_sort[idx]["mask_id"]} (sorted idx {idx})')
        ax.set_yticks([])
        ax.tick_params(axis=u'both', which=u'both', length=0)
        ax.set_xticks((0.5, 78.8), (0, 4), fontsize=20, rotation=0)
        ax.set_ylabel('Trial no.', fontsize=20, labelpad=-3)
        ax.set_xlabel('Track position [m]', fontsize=20, labelpad=-20)

        cbar = ax.collections[0].colorbar
        # cbar.ax.set_yticks((0.5, 14.5), (0, 15), fontsize=20)
        cbar.ax.tick_params(axis=u'both', which=u'both', length=0)
        cbar.ax.set_ylabel('Firing rate [Hz]', fontsize=20, labelpad=-3, rotation=270)

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
    ax.set_yticks((0.9, ex_spikerate.shape[1]-0.5), (1, ex_spikerate.shape[1]), fontsize=20, rotation=0)
    ax.set_ylabel('Trial no.', fontsize=20, labelpad=-25)
    ax.set_xlabel('Track position [m]', fontsize=20, labelpad=-20)

    cbar = ax.collections[0].colorbar
    cbar.ax.set_yticks((0.5, 18), (0, 18), fontsize=20)
    # cbar.ax.set_yticks((0, 2), (0, 2), fontsize=20)
    cbar.ax.tick_params(axis=u'both', which=u'both', length=0)
    cbar.ax.set_ylabel('Firing rate [Hz]', fontsize=20, labelpad=-5, rotation=270)
    # cbar.ax.set_ylabel(r'$\Delta$F/F', fontsize=20, labelpad=3, rotation=270)

    plt.savefig(os.path.join(folder, 'example_placecell_M69_20220228_370_fr_heatmap_1.svg'), transparent=True)

    ################################################
    # Plot single-trial spatial maps as lineplots
    stepsize = 1
    fig, ax = plt.subplots(1, 1)

    for i in range(1, ex_spikerate.shape[1]+1):
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
    ax.set_yticks((ex_spikerate.shape[1]+0.8, 1.7), (1, ex_spikerate.shape[1]), fontsize=20, rotation=0)
    ax.set_ylabel('Trial no.', fontsize=20, labelpad=-25)
    ax.set_xlabel('Track position [m]', fontsize=20, labelpad=-20)

    plt.savefig(os.path.join(folder, 'example_placecell_M69_20220228_370_dff_lines.svg'), transparent=True)

    ##################
    # Plot raw traces
    dff = (common_img.Segmentation.ROI & example_pc_keys).fetch1('dff')
    running_masks, bin_frame_counts = (hheise_placecell.Synchronization.VRTrial & example_pc_keys).fetch('running_mask', 'aligned_frames')
    n_bins, trial_mask = (hheise_placecell.PCAnalysis & example_pc_keys).fetch1('n_bins', 'trial_mask')

    # Split trace up into trials
    dff_trial = [dff[trial_mask == trial] for trial in np.unique(trial_mask)]

    # Get VR positions at frame times
    raw_pos = (hheise_behav.VRSession.VRTrial & example_pc_keys).get_arrays(['pos', 'frame'])
    pos = [tr[tr[:, 2] == 1, 1]+10 for tr in raw_pos]

    stepsize = 2
    fig, ax = plt.subplots(len(dff_trial), 1, sharex='all')
    for i in range(1, len(dff_trial)+1):
        ax[-i].plot(dff_trial[-i][running_masks[-i]], color='grey')
        ax[-i].set_yticks([])
        ax[-i].spines['top'].set_visible(False)
        ax[-i].spines['left'].set_visible(False)
        ax[-i].spines['right'].set_visible(False)
        ax[-i].spines['bottom'].set_visible(False)
        ax2 = ax[-i].twinx()
        ax2.set_yticks([])
        ax2.plot(pos[-i][running_masks[-i]], color='blue')
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        if i != 1:
            ax[-i].tick_params(axis=u'both', which=u'both', length=0)
            ax2.tick_params(axis=u'both', which=u'both', length=0)

    ax[0].set_xticks((0, 900), (0, int(900/30)), fontsize=20, rotation=0)
    ax[-1].vlines(800, ymin=1, ymax=2, label='1 dFF')
    plt.savefig(os.path.join(folder, 'example_placecell_M69_20220228_370_raw_dff_lines.svg'), transparent=True)


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
        post_dates = np.unique((common_img.Segmentation*hheise_behav.VRSession.VRTrial & 'pattern="training"' & f'mouse_id={mouse}'
                      & f'day > "{surgery_day.date() + timedelta(days=1)}"').fetch('day'))
        dates = [*pre_dates, *post_dates]
        if len(dates) == 0:
            print(f'No segmentation for mouse {mouse}')
            continue
        is_pre = [*[True] * len(pre_dates), *[False] * len(post_dates)]

        # Get average performance for each session
        if mouse == 63:
            pre_dates_63 = (hheise_behav.VRPerformance & f'mouse_id={mouse}' & f'day < "{surgery_day.date()}"').fetch('day')[
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
            si = (hheise_placecell.SpatialInformation.ROI & f'mouse_id={mouse}' & 'corridor_type=0' & f'place_cell_id=2' &
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
                data_dict = dict(mouse_id=[mouse] * len(perf), date=['2021-03-01']*5 + dates, is_pre=[True]*5 +is_pre, performance=perf,
                                 mean_fr=[np.nan]*5 +mean_fr,
                                 median_fr=[np.nan]*5 +median_fr, pc_si=[np.nan]*5 +list(pc_si), mean_stab=[np.nan]*5 +mean_stab, median_stab=[np.nan]*5 +median_stab,
                                 sd_stab=[np.nan]*5 +sd_stab, pc_bartos=[np.nan]*5 +list(pc_bartos), mean_si=[np.nan]*5 +mean_si, median_si=[np.nan]*5 +median_si)
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
                new_df[metric] = np.nanmean(data_group[(data_group['mouse_id'] == mouse) & (data_group['is_pre'] == timepoint)][metric])
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
    data.to_csv(os.path.join(folder, 'neur_perf_corr.csv'), sep='\t',
                index=False,
                columns=['mouse_id', 'performance', 'pc_si', 'mean_stab'])


#%% Single-cell plots

def filter_matched_data(match_list, keep_nomatch=False):
    """
    Filters output from get_matched_data of multiple mice/networks. Stacks cells from many networks into one array,
    removes cells that did not exist in every session.

    Args:
        match_list: List of dicts (one dict per query), each key:val pair in the dict being one network.
        keep_nomatch: Bool flag whether to keep neurons with "no-match", or remove them to only keep neurons that
                    exist in all sessions

    Returns:
        A numpy array with shape (n_total_cells, n_sessions) and possible additional dimensions, depending on input.
    """

    data_array = []
    mouse_id_list = []
    for curr_data in match_list:
        for key, net in curr_data.items():
            data_array.append(net[0])
            mouse_id_list.extend([int(key.split('_')[0])] * net[0].shape[0])
    data_array = np.vstack(data_array)
    mouse_id_list = np.array(mouse_id_list)

    # Only keep cells that exist in all sessions
    # Reduce array dimensions in case of more than 2 dimensions
    if len(data_array.shape) > 2:
        data_array_flat = np.reshape(data_array, (data_array.shape[0], data_array.shape[1] * data_array.shape[2]))
    else:
        data_array_flat = data_array

    if not keep_nomatch:
        data_array = data_array[~np.isnan(data_array_flat).any(axis=1)]
        mouse_id_list = mouse_id_list[~np.isnan(data_array_flat).any(axis=1)]

    return data_array, mouse_id_list

# Construct query to include only intended matched cells
queries = ((common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=33' & 'day<="2020-08-30"'),
           # (common_match.MatchedIndex & 'mouse_id=38' & 'day<="2020-08-24"'),
           # (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=41' & 'day<="2020-08-30"'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=108' & 'day<="2022-08-18"'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=110' & 'day<="2022-08-15"'))

spatial_maps = []
for query in queries:
    spatial_map = query.get_matched_data(table=hheise_placecell.BinnedActivity.ROI, attribute='bin_spikerate',
                                         extra_restriction=dict(corridor_type=0, place_cell_id=2),
                                         return_array=True, relative_dates=True, surgery='Microsphere injection')
    spatial_maps.append(spatial_map)

spatial_dff_maps = []
for query in queries:
    spatial_dff_map = query.get_matched_data(table=hheise_placecell.BinnedActivity.ROI, attribute='bin_activity',
                                         extra_restriction=dict(corridor_type=0, place_cell_id=2),
                                         return_array=True, relative_dates=True, surgery='Microsphere injection')
    spatial_dff_maps.append(spatial_dff_map)

is_pc = []
for query in queries:
    is_placecell = query.get_matched_data(table=hheise_placecell.PlaceCell.ROI, attribute='is_place_cell',
                                          extra_restriction=dict(corridor_type=0, place_cell_id=2),
                                          return_array=True, relative_dates=True, surgery='Microsphere injection')
    is_pc.append(is_placecell)

pfs = []
for query in queries:
    is_pf = query.get_matched_data(table=hheise_placecell.PlaceCell.PlaceField, attribute='bin_idx',
                                          extra_restriction=dict(corridor_type=0, place_cell_id=2),
                                          return_array=False, relative_dates=True, surgery='Microsphere injection')
    pfs.append(is_pf)

dffs = []
for query in queries:
    dffs.append(query.get_matched_data(table=common_img.Segmentation.ROI, attribute='dff',
                                 extra_restriction=dict(corridor_type=0),
                                 return_array=False, relative_dates=True, surgery='Microsphere injection'))


#%% Compute correlation of each neuron across days
# map_data = spatial_map['110_1'][0]
# map_data = spatial_dff_map['110_1'][0]

# Combine data from many mice/networks
map_data, mouse_idx = filter_matched_data(spatial_maps)
placecell_data, mouse_idx = filter_matched_data(is_pc)
placefield_data, mouse_idx = filter_matched_data(pfs, keep_nomatch=True)
map_dff_data, mouse_idx = filter_matched_data(spatial_dff_maps)
dff_data, mouse_idx = filter_matched_data(dffs, keep_nomatch=True)

# Smooth spatial trace
map_data = gaussian_filter1d(map_data, sigma=1, axis=2)

pearson = []
pearson_df = []
for cell_id in range(map_data.shape[0]):

    # For each neuron, compute the correlation between all session combinations
    corrmat = np.tril(np.corrcoef(map_data[cell_id]), k=-1)

    # Distinguish neighbouring and non-neighbouring days, and check PC status
    curr_pc = placecell_data[cell_id]
    neighbours = []
    pair_ids = []
    coefs = []
    where_pc = []
    for pair in combinations(np.arange(map_data.shape[1]), 2):
        if np.abs(pair[0]-pair[1]) == 1:
            neighbours.append(True)
        else:
            neighbours.append(False)

        if curr_pc[pair[0]] == curr_pc[pair[1]]:
            if curr_pc[pair[0]] == 1:
                where_pc.append('both')
            else:
                where_pc.append('none')
        elif curr_pc[pair[0]] == 1:
            where_pc.append('a')
        else:
            where_pc.append('b')

        pair_ids.append(f'd{pair[0]+1}-d{pair[1]+1}')
        coefs.append(corrmat[pair[1], pair[0]])

    # Make Fisher-Z-transform, average coefficients, then transform back for mean correlation coefficient
    # -> Average correlation of a neuron across all sessions
    pearson.append(np.tanh(np.mean(np.arctanh(coefs))))

    pearson_df.append(pd.DataFrame(data=dict(mouse=mouse_idx[cell_id], cell_id=cell_id, days=pair_ids,
                                             r=coefs, consec_days=neighbours, is_pc=where_pc)))
pearson_df = pd.concat(pearson_df, ignore_index=True)
pearson_df['days'] = pd.Categorical(pearson_df['days'],
                                    categories=['d1-d2', 'd2-d3', 'd3-d4', 'd4-d5'],
                                    ordered=True)

# Only include cells that were place cells in both comparisons
both_pc = pearson_df[pearson_df['is_pc'] == 'both']
# Only include neighbouring days
both_pc_consec = both_pc[both_pc['consec_days']]

plt.figure()
# sns.boxplot(data=both_pc_consec, x='days', y='r', hue='mouse')
ax = sns.violinplot(data=both_pc_consec, x='days', y='r')
plt.axhline(0, linestyle='--', color='grey')

plt.figure()
ax = sns.lineplot(data=both_pc_consec, x='days', y='r', )
ax.set(ylim=(0, 1))

# plt.figure()
fig, ax = plt.subplots(2, 1)
sns.barplot(data=pearson_df, x='is_pc', y='r', ax=ax[0])

plt.figure()
sns.violinplot(data=pearson_df, x='is_pc', y='r')

plt.figure()
sns.violinplot(data=pearson_df, x='is_pc', y='r', hue='mouse')

# Does place cell status affect correlation between two cells?
fig = plt.figure()
plt.axhline(0, c='grey', linestyle='--', alpha=0.5)
ax = sns.violinplot(data=pearson_df, x='is_pc', y='r')


ax_kde = sns.kdeplot(data=pearson_df, x='r', hue='is_pc', common_norm=False, common_grid=False)
# ax_kde.set_ylabel('# neurons', fontsize=25)
# ax_kde.set_xlabel('Mean correlation', fontsize=25)
# ax_kde.tick_params(axis='both', which='major', labelsize=20)
# plt.setp(ax_kde.get_legend().get_texts(), fontsize=20)
# plt.setp(ax_kde.get_legend().get_title(), fontsize=20)
ax_kde.spines['right'].set_visible(False)
ax_kde.spines['top'].set_visible(False)

# sns.stripplot(pearson)


#%% Plot cells across sessions based on sorting of reference session (where all are place cells)
def plot_matched_cells_across_sessions(traces: np.ndarray, sort_session: int, place_cells: Optional[np.ndarray] = None,
                                       normalize: bool = True, across_sessions: bool=False, smooth: Optional[int]=None):
    """
    Plot traces of matched neurons across sessions. Neurons are sorted by location of max activity in a given session.
    Args:
        traces: 3D array of the traces, shape (n_neurons, n_sessions, n_bins).
        sort_session: Index of the session where the sorting should be based on.
        place_cells: Optional, 2D array with shape (n_neurons, n_sessions). If given, only plot cells that are place
                cells in the sorted session.
        normalize: Bool Flag whether the activity should be normalized for each neuron.
        across_sessions: Bool Flag, if normalize=True, whether neuron activity should be normalized across sessions or
                or for each session separately.
        smooth: Bool Flag whether the activity should be smoothed by a Gaussian kernel of sigma=1.
    """

    if place_cells is not None:
        # Only plot place cells
        traces_filt = traces[place_cells[:, sort_session] == 1]
    else:
        # Plot all cells
        traces_filt = traces

    sort_key = [(i, np.argmax(trace)) for i, trace in enumerate(traces_filt[:, sort_session, :])]
    sort_key = [y[0] for y in sorted(sort_key, key=lambda x: x[1])]
    traces_sort = traces_filt[sort_key]

    if smooth is not None:
        traces_sort = gaussian_filter1d(traces_sort, sigma=smooth, axis=2)

    # Normalize activity neuron-wise for each sessions
    if normalize:
        if across_sessions:
            sess_squeeze = np.reshape(traces_sort, (traces_sort.shape[0], traces_sort.shape[1]*traces_sort.shape[2]))
            neur_max = np.nanmax(sess_squeeze, axis=1)
            neur_min = np.nanmin(sess_squeeze, axis=1)
            to_plot = (traces_sort - neur_min[:, None, None]) / (neur_max[:, None, None] - neur_min[:, None, None])
        else:
            traces_norm = []
            for i in range(traces_sort.shape[1]):
                neur_sess_max = np.nanmax(traces_sort[:, i, :], axis=1)
                neur_sess_min = np.nanmin(traces_sort[:, i, :], axis=1)
                traces_norm.append((traces_sort[:, i, :] - neur_sess_min[:, None]) /
                                   (neur_sess_max[:, None] - neur_sess_min[:, None]))
            to_plot = np.stack(traces_norm, axis=1)
        vmax = 1
    else:
        to_plot = traces_sort
        vmax = None

    fig, axes = plt.subplots(nrows=1, ncols=traces_sort.shape[1], figsize=(15, 7.3))
    for idx, ax in enumerate(axes):
        sns.heatmap(to_plot[:, idx, :], cmap='jet', ax=ax, cbar=False, vmax=vmax)

        # Formatting
        if idx != 0:
            ax.set_yticks([])
        else:
            ax.set_ylabel('Cell no.', fontsize=15, labelpad=-3)

        ax.set_xticks([])
        ax.set_xlabel('Track position [m]', fontsize=15)
        # ax.set_title(f'{traces_sort.shape[1]-idx}d prestroke')

    plt.tight_layout()


# Combine data from many mice/networks
map_data, mouse_idx = filter_matched_data(spatial_maps)
placecell_data, mouse_idx = filter_matched_data(is_pc)
map_dff_data, mouse_idx = filter_matched_data(spatial_dff_maps)

# plot_matched_cells_across_sessions(map_data, 2, place_cells=placecell_data, normalize=True)
plot_matched_cells_across_sessions(traces=map_dff_data, sort_session=4, place_cells=placecell_data, normalize=True,
                                   across_sessions=True, smooth=1)

#%% Place Cell/Place field qualitative analysis

def place_cell_qualitative():
    """
    Check which place cells are also place cells in other sessions with the same/similar place field (stable place
    cells), and which cells are place cells, but at a different place in the corridor (remapping place cells), or
    are not place cells anymore.
    """

    # Construct query to include only intended matched cells
    queries = ((common_match.MatchedIndex & 'mouse_id=33' & 'day<="2020-08-24"'),
               # (common_match.MatchedIndex & 'mouse_id=38' & 'day<="2020-08-24"'),
               (common_match.MatchedIndex & 'mouse_id=41' & 'day<="2020-08-24"'),
               (common_match.MatchedIndex & 'mouse_id=108' & 'day<="2022-08-12"'),
               (common_match.MatchedIndex & 'mouse_id=110' & 'day<="2022-08-09"'))

    is_pc = []
    pfs = []
    spatial_maps = []
    match_matrices = []
    for query in queries:
        match_matrices.append(query.construct_matrix())

        is_pc.append(query.get_matched_data(table=hheise_placecell.PlaceCell.ROI, attribute='is_place_cell',
                                            extra_restriction=dict(corridor_type=0, place_cell_id=2),
                                            return_array=True, relative_dates=True, surgery='Microsphere injection'))

        pfs.append(query.get_matched_data(table=hheise_placecell.PlaceCell.PlaceField, attribute='bin_idx',
                                          extra_restriction=dict(corridor_type=0, place_cell_id=2),
                                          return_array=False, relative_dates=True, surgery='Microsphere injection'))

        spatial_maps.append(query.get_matched_data(table=hheise_placecell.BinnedActivity.ROI, attribute='bin_spikerate',
                                             extra_restriction=dict(corridor_type=0, place_cell_id=2),
                                             return_array=True, relative_dates=True, surgery='Microsphere injection'))

    placecell_data, mouse_idx = filter_matched_data(is_pc, keep_nomatch=True)
    placefield_data, mouse_idx = filter_matched_data(pfs, keep_nomatch=True)
    spatial_data, mouse_idx = filter_matched_data(spatial_maps, keep_nomatch=True)

    # Get matrix with mask IDs
    match_matrices = []
    for query in queries:
        match_matrices.append(query.construct_matrix())

    # For all place cells on a specific day, how many are also place cells on other days?
    pc_d5 = placecell_data[placecell_data[:, 4] == 1]
    print(np.nansum(pc_d5, axis=0))

    # How many place cells are still place cells on the next day?
    mice_data = []
    for mouse in np.unique(mouse_idx):
        curr_mouse = placecell_data[mouse_idx == mouse]
        print(f'\nMouse {mouse}:')
        mouse_data = []
        for i in range(curr_mouse.shape[1]-1):
            curr_pc = curr_mouse[curr_mouse[:, i] == 1]
            print(f'Day {i}: {len(curr_pc)} Place Cells, {np.nansum(curr_pc[:, i+1])} place cells on day {i+1} '
                  f'({(np.nansum(curr_pc[:, i+1]) / len(curr_pc))*100})')
            mouse_data.append((np.nansum(curr_pc[:, i+1]) / len(curr_pc))*100)
        mice_data.append(np.array(mouse_data))
    mice_data = np.stack(mice_data)

    np.nanmean(mice_data[[True, False, True, True, True]], axis=0)

    # Of these cells, how many have the place field in approximately the same position?
    pfs = []
    for query in queries:
        pfs.append(query.get_matched_data(table=hheise_placecell.PlaceCell.PlaceField, attribute='bin_idx',
                                          extra_restriction=dict(corridor_type=0, place_cell_id=2),
                                          return_array=False, relative_dates=True, surgery='Microsphere injection'))

    placefield_data, mouse_idx = filter_matched_data(pfs, keep_nomatch=True)

    pf_data = []

    def place_field_com(spat_map, pf_list):
        """
        Compute center of mass for place fields of a single cell.

        Args:
            spat_map: 1D array, spatial activity map of one cell
            pf_list: List of arrays holding indices of one or more place fields

        Returns:

        """

        for place_field in pf_list:
            curr_pf = spat_map[place_field]

        # Normalize
        map_norm = (spat_map - np.min(spat_map)) / (np.max(spat_map) - np.min(spat_map))
        # Convert to Probability Mass Function / Probability distribution
        map_pmf = map_norm / np.sum(map_norm)
        # Calculate moments
        com = np.sum(np.arange(len(map_pmf)) * map_pmf)

        com_std = []
        for t in np.arange(len(map_pmf)):
            com_std.append((t**2 * map_pmf[t]) - com**2)
        com_std = np.sqrt(np.sum(np.arange(len(map_pmf))**2 * map_pmf) - com**2)

        plt.plot(spat_map)
        plt.axvline(com, color='r')
        plt.axvspan(com-com_std/2, com+com_std/2, color='r', alpha=0.3)

    for i in range(placecell_data.shape[1]-1):
        # Idx of PCs in session i that are also PCs in session i+1
        stable_pc_idx = np.where(np.nansum(placecell_data[:, i:i+2], axis=1) == 2)[0]
        pfs_1 = placefield_data[stable_pc_idx, i]
        pfs_2 = placefield_data[stable_pc_idx, i+1]

        # For each stable place cell, compare place fields
        for cell_idx, pf_1, pf_2 in zip(stable_pc_idx, pfs_1, pfs_2):


            cell_idx = 26
            mask_id = match_matrices[0]['33_1'].iloc[cell_idx, i]
            cell_1 = dict(username='hheise', mouse_id=33, day='2020-08-18', mask_id=mask_id, place_cell_id=2)

            test_map = np.mean((hheise_placecell.BinnedActivity.ROI & cell_1).fetch1('bin_spikerate'), axis=1)
            spat_map = spatial_data[cell_idx, i]
            spat_map1 = spatial_maps[0]['33_1'][0][cell_idx, i]

            (hheise_placecell.PlaceCell.PlaceField & cell_1).fetch('bin_idx')
            curr_pf = placefield_data[cell_idx, i]

            cell_2 = dict(username='hheise', mouse_id=33, day='2020-08-19', mask_id=294, place_cell_id=2)





            # Get center of mass for current place fields in both sessions
            place_field_com(spat_map=map_data[cell_idx, i], pf_list=pf_1)



        mouse_data.append((np.nansum(curr_pc[:, i+1]) / len(curr_pc))*100)
    mice_data.append(np.array(mouse_data))