#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 15/12/2022 12:07
@author: hheise

Plots for the preprint December 2022
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
             83, 85, 86, 89, 90, 91, 93, 95,  # Batch 7
             108, 110, 111, 112, 113, 114, 115, 116, 121, 122]  # Batch 8

#%% Different ways of grouping

# Old grouping (behavior-based, with flicker)
no_deficit = [93, 91, 95]
no_deficit_flicker = [111, 114, 116]
recovery = [33, 83, 85, 86, 89, 90, 113]
deficit_no_flicker = [41, 63, 69, 121]
deficit_flicker = [108, 110, 112]
sham_injection = [115, 122]

# Split sham (no spheres) and no_deficit (no deficit, but > 50 spheres), based on bin-lick-ratio (except 93 and 69 on SI, ignore flicker)
sham = [33, 91, 115, 122, 111]               # no deficit, and < 50 spheres
no_deficit = [83, 95, 108, 112, 114, 116, 121]   # no deficit, but > 50 spheres (merge with sham if spheres should be ignored)
late_deficit = [93]                          # Debatable, might be merged with sham
recovery = [85, 86, 89, 90, 113]
no_recovery = [41, 63, 69, 110]

# Split sham (no spheres) and no_deficit (no deficit, but > 50 spheres), based on SI (ignore flicker)
sham = [91, 115, 122]                  # no deficit, and < 50 spheres
no_deficit = [83, 95, 112, 114, 116]   # no deficit, but > 50 spheres
late_deficit = [111, 93, 108]                          # Debatable, might be merged with sham
recovery = [33, 85, 86, 90, 113]
no_recovery = [41, 63, 69, 89, 110, 121]

# Grouping by number of spheres (extrapolated for whole brain)
low = [38, 91, 111, 115, 122]   # < 50 spheres
mid = [33, 83, 86, 89, 93, 95, 108, 110, 112, 113, 114, 116, 121]  # 50 - 500 spheres
high = [41, 63, 69, 85, 90, 109, 123]   # > 500 spheres

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
        days = (hheise_behav.VRPerformance & 'username="hheise"' & f'mouse_id={mouse}' & f'day<="{surgery_day}"').fetch('day')
        perf_lick = (hheise_behav.VRPerformance & 'username="hheise"' & f'mouse_id={mouse}' & f'day<="{surgery_day}"').get_mean('binned_lick_ratio')
        perf_si = (hheise_behav.VRPerformance & 'username="hheise"' & f'mouse_id={mouse}' & f'day<="{surgery_day}"').get_mean('si_binned_run')
        perf_dist = (hheise_behav.VRPerformance & 'username="hheise"' & f'mouse_id={mouse}' & f'day<="{surgery_day}"').get_mean('distance')

        # Transform dates into days before surgery
        rel_days = [(d - surgery_day).days for d in days]
        # Get session number before surgery
        rel_sess = np.arange(-len(perf_lick), 0)

        df_wide = pd.DataFrame(dict(mouse_id=mouse, days=days, rel_days=rel_days, rel_sess=rel_sess, blr=perf_lick,
                                    si=perf_si, dist=perf_dist))
        dfs.append(df_wide.melt(id_vars=['mouse_id', 'days', 'rel_days', 'rel_sess'], var_name='metric',
                                value_name='perf'))
    df = pd.concat(dfs, ignore_index=True)

    sns.lineplot(data=df, x='rel_sess', y='perf', hue='metric')

    # Export for prism
    for metric in ['blr', 'si', 'dist']:
        df_exp = df[df['metric']==metric].pivot(index='rel_sess', columns='mouse_id', values='perf')
        df_exp.to_csv(os.path.join(folder, 'figure1\\learning_curve', f'learning_curve_{metric}.csv'), sep=',')


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
    ax = sns.heatmap(gaussian_filter1d(avg_act_filt_sort[sparse_neuron_mask], sigma=1, axis=1), cmap='jet',
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
    plt.savefig(os.path.join(folder, 'figure1', 'all_place_cells_bartos_8020_for_matrix_jet.png'), transparent=True)
    # Much fewer cells, this file is used to load quickly into Inkscape, delete the matrix and use the axes
    plt.savefig(os.path.join(folder, 'figure1', 'all_place_cells_3380_for_axes.svg'), transparent=True)


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
        long_sess = [pk[i] for i in range(len(norm_trials)) if len(norm_trials[i]) >= 10]

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
        ax = sns.heatmap(gaussian_filter1d(act_sort[idx].T, sigma=1, axis=1), cmap='jet')  # Cap colormap a bit
        # ax = sns.heatmap(avg_act_sort_norm, cmap='jet')

        # Shade reward zones
        zone_borders = (hheise_behav.CorridorPattern & 'pattern="training"').rescale_borders(n_bins=80)
        for zone in zone_borders:
            ax.axvspan(zone[0], zone[1], facecolor='green', alpha=0.4)

        # Clean up axes and color bar
        ax.set_title(
            f'M{pk_sort[idx]["mouse_id"]}_{pk_sort[idx]["day"]}_mask{pk_sort[idx]["mask_id"]} (sorted idx {idx})')
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

#%% Figure 2
# Sphere count behavior groups
spheres = pd.DataFrame((hheise_hist.MicrosphereSummary.Metric &
                        'metric_name="spheres"').fetch('mouse_id','count_extrap', as_dict=True))
merge = spheres.merge(grouping_2, on='mouse_id')
merge = merge.rename(columns={'count_extrap': 'sphere count'})

plt.figure()
sns.set(font_scale=1.25)
strp = sns.stripplot(merge, y='group', x='sphere count', size=12, orient='h',
                     order=['no_deficit', 'recovery', 'deficit_flicker', 'deficit_no_flicker'])
# strp.set_xscale('log')
label = [10, 50, 100, 200, 500, 1000, 2000]
strp.set(xticks=label, xticklabels=label)
for i, point in enumerate(strp.collections):
    # Extract the x and y coordinates of the data point
    x = point.get_offsets()[:, 1]
    y = point.get_offsets()[:, 0]

    # Add labels to the data point
    for j, y_ in enumerate(y):
        plt.text(y[j], x[j]-0.05, merge[merge['sphere count'] == y_]['mouse_id'].iloc[0], ha='center', va='bottom',
                 fontsize=15)

### Behavior curve after stroke groups
dfs = []
for mouse in mice:
    # Get day of surgery
    surgery_day = (common_mice.Surgery() & 'surgery_type="Microsphere injection"' & f'mouse_id={mouse}').fetch(
        'surgery_date')[0].date()
    # Get date and performance for each session before the surgery day
    days = (hheise_behav.VRPerformance & 'username="hheise"' & f'mouse_id={mouse}').fetch('day')
    perf_lick = (hheise_behav.VRPerformance & 'username="hheise"' & f'mouse_id={mouse}').get_mean('binned_lick_ratio')
    perf_si = (hheise_behav.VRPerformance & 'username="hheise"' & f'mouse_id={mouse}').get_mean('si_binned_run')
    perf_dist = (hheise_behav.VRPerformance & 'username="hheise"' & f'mouse_id={mouse}').get_mean('distance')

    # Transform dates into days before surgery
    rel_days = [(d - surgery_day).days for d in days]
    # Get session number before surgery
    rel_sess = np.arange(-len(perf_lick), 0)

    df_wide = pd.DataFrame(dict(mouse_id=mouse, days=days, rel_days=rel_days, rel_sess=rel_sess, blr=perf_lick,
                                si=perf_si, dist=perf_dist))
    dfs.append(df_wide.melt(id_vars=['mouse_id', 'days', 'rel_days', 'rel_sess'], var_name='metric',
                            value_name='perf'))
df = pd.concat(dfs, ignore_index=True)

sns.lineplot(data=df, x='rel_sess', y='perf', hue='metric')

# Export for prism
for metric in ['blr', 'si', 'dist']:
    df_exp = df[df['metric'] == metric].pivot(index='rel_days', columns='mouse_id', values='perf')
    df_exp.to_csv(os.path.join(folder, 'figure2', f'performance_over_time_{metric}.csv'), sep=',')


#%% Figure 3

# Place Cells per FOV over time
pc_ratios = pd.DataFrame((hheise_placecell.PlaceCell() & 'place_cell_id=2' & f'mouse_id in {helper.in_query(mice)}' &
                          'corridor_type=0').fetch('KEY', 'place_cell_ratio', as_dict=True))
pc_ratios['place_cell_ratio'] *= 100

rel_days = []
for i, row in pc_ratios.iterrows():
    surg_day = (common_mice.Surgery & f'mouse_id={row["mouse_id"]}' &
                'surgery_type="Microsphere injection"').fetch1('surgery_date')
    rel_days.append((row['day'] - surg_day.date()).days)
pc_ratios['rel_days'] = rel_days

g = sns.FacetGrid(data=pc_ratios, col='mouse_id', col_wrap=6, sharex=False, sharey=False)
g.map_dataframe(sns.lineplot, x='rel_days', y='place_cell_ratio')
for col_val, ax in g.axes_dict.items():
    ax.axvline(0.5, linestyle='--', c='r')


def compute_pc_ratio_change(pc_df, relative_diff=False):
    """ To be applied to a DF from a single mouse via groupby(). """

    three_day_pre = np.sort(pc_df[pc_df['rel_days'] <= 0]['rel_days'].values)[-3]

    pre_avg = np.nanmean(pc_df[pc_df['rel_days'].between(three_day_pre, 0)]['place_cell_ratio'])

    if relative_diff:
        out = pd.DataFrame([dict(
            early_post_avg=(np.nanmean(pc_df[pc_df['rel_days'].between(1, 9)]['place_cell_ratio']) / pre_avg) * 100,
            late_post_avg=(np.nanmean(pc_df[pc_df['rel_days'] > 9]['place_cell_ratio']) / pre_avg) * 100,
            total_post_avg=(np.nanmean(pc_df[pc_df['rel_days'] > 0]['place_cell_ratio']) / pre_avg) * 100
        )])
    else:
        out = pd.DataFrame([dict(
            early_post_avg=np.nanmean(pc_df[pc_df['rel_days'].between(1, 9)]['place_cell_ratio']) - pre_avg,
            late_post_avg=np.nanmean(pc_df[pc_df['rel_days'] > 9]['place_cell_ratio']) - pre_avg,
            total_post_avg=np.nanmean(pc_df[pc_df['rel_days'] > 0]['place_cell_ratio']) - pre_avg
        )])
    return out


pc_abs_change = pc_ratios.groupby('mouse_id').apply(compute_pc_ratio_change).droplevel(1)
pc_rel_change = pc_ratios.groupby('mouse_id').apply(compute_pc_ratio_change, relative_diff=True).droplevel(1)

spheres = pd.DataFrame((hheise_hist.MicrosphereSummary.Metric &
                        'metric_name="spheres"').fetch('mouse_id', 'count_extrap', as_dict=True))
spheres.rename(columns={'count_extrap': 'sphere_count'}, inplace=True)

pc_abs = pc_abs_change.merge(spheres, left_index=True, right_on='mouse_id')
pc_rel = pc_rel_change.merge(spheres, left_index=True, right_on='mouse_id')
pc_abs.to_csv(r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Papers\preprint\figure3\abs_pc_ratio_change.csv')
pc_rel.to_csv(r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Papers\preprint\figure3\rel_pc_ratio_change.csv')

pc_abs_melt = pc_abs.melt(id_vars=['mouse_id', 'sphere_count'], var_name='time_point', value_name='abs_pc_diff')
pc_rel_melt = pc_rel.melt(id_vars=['mouse_id', 'sphere_count'], var_name='time_point', value_name='rel_pc_diff')

ax = sns.scatterplot(data=pc_abs_melt, x='sphere_count', y='abs_pc_diff', hue='time_point')
ax.set_xscale('log')

# Correlate metrics (PC ratio, within-session stability, firing rate) with behavioral performance
stab = pd.DataFrame((hheise_placecell.SpatialInformation.ROI & 'place_cell_id=2' & f'mouse_id in {helper.in_query(mice)}' &
                     'corridor_type=0').fetch('mouse_id', 'day', 'stability', as_dict=True))
stab_avg = stab.groupby(['mouse_id', 'day']).mean().reset_index()

# Firing rate (only normal trials)
pks = (common_img.Deconvolution & f'mouse_id in {helper.in_query(mice)}' & 'decon_id=1').fetch('KEY')
fr_avg = []
for pk in pks:
    spikes = (common_img.Segmentation & pk).get_traces('decon', decon_id=pk['decon_id'])
    fps = (common_img.ScanInfo & pk).fetch1('fr')
    trial_mask = (hheise_placecell.PCAnalysis & pk & 'place_cell_id=2').fetch1('trial_mask')
    normal_trials = (hheise_behav.VRSession & pk).get_normal_trials()
    normal_trial_mask = np.isin(trial_mask, normal_trials)
    spikes = spikes[:, normal_trial_mask]
    fr_avg.append(pd.DataFrame([dict(mouse_id=pk['mouse_id'], day=pk['day'],
                                     avg_fr=np.mean(np.sum(spikes, axis=1) / spikes.shape[1] * fps))]))
fr_avg = pd.concat(fr_avg)

#%% Single-cell plots

# Construct query to include only intended matched cells
queries = ((common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=33' & 'day<="2020-08-30"'),
           # (common_match.MatchedIndex & 'mouse_id=38' & 'day<="2020-08-24"'),
           # (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=41' & 'day<="2020-08-30"'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=108' & 'day<"2022-08-18"'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=110' & 'day<"2022-08-15"'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=121' & 'day<"2022-08-15"'))

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
decons = []
for query in queries:
    decons.append(query.get_matched_data(table=common_img.Deconvolution.ROI, attribute='decon',
                                       extra_restriction=dict(corridor_type=0),
                                       return_array=False, relative_dates=True, surgery='Microsphere injection'))

# %% Compute correlation of each neuron across days
# map_data = spatial_map['110_1'][0]
# map_data = spatial_dff_map['110_1'][0]

# Combine data from many mice/networks
map_data, mouse_idx = dc.filter_matched_data(spatial_maps)
placecell_data, mouse_idx = dc.filter_matched_data(is_pc)
placefield_data, mouse_idx = dc.filter_matched_data(pfs, keep_nomatch=True)
map_dff_data, mouse_idx = dc.filter_matched_data(spatial_dff_maps)
dff_data, mouse_idx = dc.filter_matched_data(dffs, keep_nomatch=True)

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
        if np.abs(pair[0] - pair[1]) == 1:
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

        pair_ids.append(f'd{pair[0] + 1}-d{pair[1] + 1}')
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


# %% Plot cells across sessions based on sorting of reference session (where all are place cells)
def plot_matched_cells_across_sessions(traces: np.ndarray, sort_session: int, place_cells: Optional[np.ndarray] = None,
                                       normalize: bool = True, across_sessions: bool = False,
                                       titles: Optional[Iterable] = None,
                                       smooth: Optional[int] = None, cmap='turbo'):
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
        titles: List of titles for each subplot/session.
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
            sess_squeeze = np.reshape(traces_sort, (traces_sort.shape[0], traces_sort.shape[1] * traces_sort.shape[2]))
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
        sns.heatmap(to_plot[:, idx, :], cmap=cmap, ax=ax, cbar=False, vmax=vmax)

        # Formatting
        if idx != 0:
            ax.set_yticks([])
        else:
            ax.set_ylabel('Cell no.', fontsize=15, labelpad=-3)

        ax.set_xticks([])
        ax.set_xlabel('Track position [m]', fontsize=15)
        # ax.set_title(f'{traces_sort.shape[1]-idx}d prestroke')

        if titles is not None:
            ax.set_title(titles[idx])

    plt.tight_layout()

    return fig

# Combine data from many mice/networks
map_data, mouse_idx = dc.filter_matched_data(spatial_maps)
placecell_data, mouse_idx = dc.filter_matched_data(is_pc)
map_dff_data, mouse_idx = dc.filter_matched_data(spatial_dff_maps)


# plot_matched_cells_across_sessions(map_data, 2, place_cells=placecell_data, normalize=True)
plot_matched_cells_across_sessions(traces=map_dff_data, sort_session=4, place_cells=placecell_data, normalize=True,
                                   across_sessions=True, smooth=1)

# Plot highly correlating and low correlating cells separately across days
pearson_df_consec = pearson_df[pearson_df['consec_days']]
cell_means = pearson_df_consec.groupby('cell_id')['r'].mean()

# Only use place cells in sorting session
sort_session = 0
pc_mask = placecell_data[:, sort_session] == 1
cell_means = cell_means[pc_mask]

upper_quantile = np.quantile(cell_means, 0.75)
lower_quantile = np.quantile(cell_means, 0.25)
plt.figure()
# bins = sns.histplot(cell_means, bins=30)
sns.kdeplot(cell_means)
plt.axvline(upper_quantile)
plt.axvline(lower_quantile)

plot_matched_cells_across_sessions(traces=map_dff_data[cell_means > upper_quantile], sort_session=sort_session,
                                   #place_cells=placecell_data[cell_means < lower_quantile],
                                   normalize=True, across_sessions=True, smooth=1, cmap='turbo')
plot_matched_cells_across_sessions(traces=map_dff_data[cell_means < lower_quantile], sort_session=sort_session,
                                   #place_cells=placecell_data[cell_means < lower_quantile],
                                   normalize=True, across_sessions=True, smooth=1)


#%% Place Cell/Place field qualitative analysis (Pie Charts)

def place_cell_qualitative():
    """
    Check which place cells are also place cells in other sessions with the same/similar place field (stable place
    cells), and which cells are place cells, but at a different place in the corridor (remapping place cells), or
    are not place cells anymore.
    """

    def get_remaining_newlycoding_cellcounts(dataset):
        """ Dataset is the output of is_pc queried for only prestroke sessions of a single network. """
        arr = list(dataset.items())[0][1][0]
        n_newly = 0
        n_remaining = 0
        for i in range(1, arr.shape[1]):
            curr_pc = arr[arr[:, i] == 1]  # Take all PCs at the current session
            # curr_pc = curr_pc[~np.isnan(curr_pc[:, i - 1])]  # Ignore PCs that were not tracked in the previous session
            n_remaining += np.nansum(curr_pc[:, i - 1])
            n_newly += len(curr_pc) - np.nansum(curr_pc[:, i - 1])

        return n_newly, n_remaining

    def get_stability_cellcounts(is_pc_dataset, placefield_dataset, spatial_dff_dataset):
        """ Dataset is the output of is_pc, pfs and spat_dff_maps of a single network. """
        pc_arr = list(is_pc_dataset.items())[0][1][0]
        pf_arr = np.array(list(placefield_dataset.items())[0][1][0])
        dff_arr = list(spatial_dff_dataset.items())[0][1][0]

        # Go through every session
        dfs = []

        # for i in range(placecell_data.shape[1] - 1):      # This checks PCs on the next day (forward)
        #     i_next = i+1
        for i in range(1, pc_arr.shape[1]):  # This checks PCs on the previous day (backward)
            i_next = i - 1

            # Idx of PCs in session i that are also PCs in the other session
            stable_pc_idx = np.where(np.nansum(pc_arr[:, [i, i_next]], axis=1) == 2)[0]
            pfs_1 = pf_arr[stable_pc_idx, i]
            pfs_2 = pf_arr[stable_pc_idx, i_next]

            # For each stable place cell, compare place fields
            for cell_idx, pf_1, pf_2 in zip(stable_pc_idx, pfs_1, pfs_2):

                # Get center of mass for current place fields in both sessions
                pf_com_1 = [dc.place_field_com(spatial_map_data=dff_arr[cell_idx, i], pf_indices=pf) for pf in pf_1]
                pf_com_2 = [dc.place_field_com(spatial_map_data=dff_arr[cell_idx, i_next], pf_indices=pf) for pf in pf_2]

                # For each place field in day i, check if there is a place field on day 2 that overlaps with its CoM
                pc_is_stable = np.nan
                dist = np.nan
                if len(pf_com_1) == len(pf_com_2):
                    # If cell has one PF on both days, check if the day-1 CoM is located inside day-2 PF -> stable
                    if len(pf_com_1) == 1:
                        if pf_2[0][0] < pf_com_1[0][0] < pf_2[0][-1]:
                            pc_is_stable = True
                        else:
                            pc_is_stable = False

                        # Compute distance for now only between singular PFs (positive distance means shift towards end)
                        dist = pf_com_2[0][0] - pf_com_1[0][0]

                    # If the cell has more than 1 PF on both days, check if all PFs overlap -> stable
                    else:
                        same_pfs = [True if pf2[0] < pf_com1[0] < pf2[-1] else False for pf_com1, pf2 in
                                    zip(pf_com_1, pf_2)]
                        pc_is_stable = all(
                            same_pfs)  # This sets pc_is_stable to True if all PFs match, otherwise its False

                # If cell has different number of place fields on both days, for now don't process
                else:
                    pass

                # Get difference in place field numbers to quantify how many cells change number of place fields
                pf_num_change = len(pf_com_2) - len(pf_com_1)

                dfs.append(pd.DataFrame([dict(mouse_id=list(is_pc_dataset.items())[0][0], cell_idx=cell_idx, day=i,
                                              day_next=i_next, stable=pc_is_stable, dist=dist, num_pf_1=len(pf_com_1),
                                              num_pf_2=len(pf_com_2), pf_num_change=pf_num_change)]))
        return pd.concat(dfs)

    # Construct query to include only intended matched cells
    queries = ((common_match.MatchedIndex & 'mouse_id=33' & 'day<="2020-08-24"'),
               # (common_match.MatchedIndex & 'mouse_id=38' & 'day<="2020-08-24"'),
               (common_match.MatchedIndex & 'mouse_id=41' & 'day<="2020-08-24"'),
               (common_match.MatchedIndex & 'mouse_id=69' & 'day<="2021-03-08"'),
               (common_match.MatchedIndex & 'mouse_id=85' & 'day<="2021-07-16"'),
               (common_match.MatchedIndex & 'mouse_id=90' & 'day<="2021-07-16"'),
               (common_match.MatchedIndex & 'mouse_id=93' & 'day<="2021-07-21"'),
               (common_match.MatchedIndex & 'mouse_id=108' & 'day<="2022-08-12"'),
               (common_match.MatchedIndex & 'mouse_id=110' & 'day<="2022-08-09"'),
               (common_match.MatchedIndex & 'mouse_id=121' & 'day<="2022-08-12"'),
               (common_match.MatchedIndex & 'mouse_id=115' & 'day<="2022-08-09"'),
               (common_match.MatchedIndex & 'mouse_id=122' & 'day<="2022-08-09"'),
               (common_match.MatchedIndex & 'mouse_id=114' & 'day<="2022-08-09"'))

    is_pc = []
    pfs = []
    spatial_maps = []
    match_matrices = []
    spat_dff_maps = []

    for query in queries:
        match_matrices.append(query.construct_matrix())

        is_pc.append(query.get_matched_data(table=hheise_placecell.PlaceCell.ROI, attribute='is_place_cell',
                                            extra_restriction=dict(corridor_type=0, place_cell_id=2),
                                            return_array=True, relative_dates=True, surgery='Microsphere injection'))

        pfs.append(query.get_matched_data(table=hheise_placecell.PlaceCell.PlaceField, attribute='bin_idx',
                                          extra_restriction=dict(corridor_type=0, place_cell_id=2),
                                          return_array=False, relative_dates=True, surgery='Microsphere injection'))

        # spatial_maps.append(query.get_matched_data(table=hheise_placecell.BinnedActivity.ROI, attribute='bin_spikerate',
        #                                            extra_restriction=dict(corridor_type=0, place_cell_id=2),
        #                                            return_array=True, relative_dates=True,
        #                                            surgery='Microsphere injection'))

        spat_dff_maps.append(query.get_matched_data(table=hheise_placecell.BinnedActivity.ROI, attribute='bin_activity',
                                                    extra_restriction=dict(corridor_type=0, place_cell_id=2),
                                                    return_array=True, relative_dates=True,
                                                    surgery='Microsphere injection'))

    # Get number of tracked cells
    n_tracked_cells = np.sum([len(list(mouse.items())[0][1]) for mouse in match_matrices])
    remain_newly = [get_remaining_newlycoding_cellcounts(mouse) for mouse in is_pc]
    remain_newly = np.array(remain_newly)
    print('Number of cells Newly coding: ', np.sum(remain_newly[:, 0]))
    print('Number of cells Remaining PCs: ', np.sum(remain_newly[:, 1]))

    ### Create array with stable/remapping PCs
    stability_dfs = [get_stability_cellcounts(pc_data, pf_data, dff_maps) for pc_data, pf_data, dff_maps in
                     zip(is_pc, pfs, spat_dff_maps)]
    stability_df = pd.concat(stability_dfs)
    print('Number of stable PCs: ', stability_df['stable'].sum())
    print('Number of remapping PCs: ', len(stability_df) - stability_df['stable'].sum())

    # Plot activity of single cell across days vertically
    """
    Examples:
    230: remaps once, then stable
    284: stable, but remapped to different RZ
    343: relatively silent, completely remaps to different location
    506: stable place cell for 1st and 3rd RZ
    561: quite stable, bit messy
    """
    cell_idx = 230
    plt.figure()
    sns.heatmap(gaussian_filter1d(spatial_dff_data[cell_idx], axis=1, sigma=1), cbar=False, cmap='turbo')

    # zone_borders = (hheise_behav.CorridorPattern & 'pattern="training"').rescale_borders(n_bins=80)
    # for zone in zone_borders:
    #     plt.axvspan(zone[0], zone[1], facecolor='green', alpha=0.4)
    plt.title(cell_idx)

# %% Plot examples for stable/remapping place cells

# Example of prolonged place field (artificial)
x = np.arange(80)
y = np.array([5.22377379e-02, 2.26253215e-02, 2.04782374e-02, 1.96897145e-02,
              2.09978614e-02, 5.29242679e-02, 3.14436592e-02, 2.93919835e-02,
              5.09827323e-02, 6.15774915e-02, 4.51521464e-02, 6.15792312e-02,
              7.02343956e-02, 2.57525612e-02, 1.21903546e-01, 1.95630357e-01,
              1.05794154e-01, 6.25002757e-02, 3.73290367e-02, 2.91241109e-02,
              1.35791358e-02, 1.19158691e-02, 1.18616242e-02, 8.81965272e-03,
              1.83015168e-02, 2.69463416e-02, 9.53717753e-02, 1.78993255e-01,
              2.69650608e-01, 2.59652168e-01, 5.59530872e-01, 8.05883515e-01,
              11.03258836e-01, 15.39960182e-01, 12.09657359e-01, 9.81399548e-01,
              8.58287406e-01, 4.18269181e-01, 3.70853513e-01, 3.34044358e-01,
              3.22030086e-01, 3.00857176e-01, 3.15183845e-01, 3.00594736e-01,
              2.82286466e-01, 2.40064371e-01, 2.34399972e-01, 2.17649590e-01,
              2.43314377e-01, 2.81106620e-01, 2.78584864e-01, 2.56415936e-01,
              1.52608334e-01, 1.77566256e-01, 1.80633550e-01, 1.10497845e-01,
              1.30287064e-01, 2.72508503e-02, 2.36224495e-02, 1.62607748e-02,
              2.59395572e-03, 4.92016645e-03, -1.00035751e-02, 3.80560267e-03,
              7.75086926e-03, 1.17135223e-03, 8.12307280e-03, -1.12709999e-02,
              1.02724927e-02, 2.21360358e-03, 1.65023524e-02, 3.30559947e-02,
              3.19987424e-02, 1.41764000e-01, 1.72215104e-01, 1.00743033e-01,
              4.65977229e-02, 1.12166762e-01, 8.03439096e-02, 4.09437418e-02])
plt.figure(figsize=(12, 7))
plt.plot(x, y)
plt.axvspan(28, 51, color='red', alpha=0.2)
plt.axvline(dc.place_field_com(spatial_map_data=y, pf_indices=np.arange(28, 51))[0], color='red')


# Stable, single PF
cell_id = 53
days = (0, 1)

# Remapping, single PF, to different location
cell_id = 421
days = (0, 1)
cell_id = 359
days = (2, 3)

# Stable, double PF
cell_id = 134
days = (1, 2)

# remapping, double PF
cell_id = 510
days = (0, 1)

# changing numbers of PF, one additional PF
cell_id = 134
days = (0, 1)
cell_id = 502
days = (1, 2)

# changing numbers of PF, completely different PFs
cell_id = 254
days = (0, 1)
cell_id = 363
days = (2, 3)

# changing numbers of PF, one additional PF at place where there was activity, but not accepted PF
cell_id = 279
days = (0, 1)
cell_id = 43
days = (1, 2)
cell_id = 254
days = (1, 2)


cell_id = 359
days = (2, 3)

# Plotting code
fig, ax = plt.subplots(2, 1, sharey='all', figsize=(12, 7))
ax[0].plot(spatial_dff_data[cell_id, days[0]])
for pf in placefield_data[cell_id, days[0]]:
    ax[0].axvspan(pf[0], pf[-1], color='red', alpha=0.2)
    ax[0].axvline(dc.place_field_com(spatial_map_data=spatial_dff_data[cell_id, days[0]], pf_indices=pf)[0], color='red')

ax[1].plot(spatial_dff_data[cell_id, days[1]])
for pf in placefield_data[cell_id, days[1]]:
    ax[1].axvspan(pf[0], pf[-1], color='red', alpha=0.2)
    ax[1].axvline(dc.place_field_com(spatial_map_data=spatial_dff_data[cell_id, days[1]], pf_indices=pf)[0], color='red')

ax[0].set_xticks([])
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)

# zone_borders = (hheise_behav.CorridorPattern & 'pattern="training"').rescale_borders(n_bins=80)
for zone in zone_borders:
    ax[0].axvspan(zone[0], zone[1], color='green', alpha=0.2)
    ax[1].axvspan(zone[0], zone[1], color='green', alpha=0.2)


#%% Pie Charts (total across mice and days)

# Place cell ratio
queries = ((hheise_placecell.PlaceCell & 'mouse_id=33' & 'day<="2020-08-24"' & 'place_cell_id=2' & 'corridor_type=0'),
           (hheise_placecell.PlaceCell & 'mouse_id=38' & 'day<="2020-08-24"' & 'place_cell_id=2' & 'corridor_type=0'),
           (hheise_placecell.PlaceCell & 'mouse_id=41' & 'day<="2020-08-24"' & 'place_cell_id=2' & 'corridor_type=0'),
           (hheise_placecell.PlaceCell & 'mouse_id=108' & 'day<="2022-08-12"' & 'place_cell_id=2' & 'corridor_type=0'),
           (hheise_placecell.PlaceCell & 'mouse_id=110' & 'day<="2022-08-09"' & 'place_cell_id=2' & 'corridor_type=0'))

pks = [query.fetch('KEY') for query in queries]
pks = [item for sublist in pks for item in sublist]

# Place cell ratio plot, split by days and mice
pc_ratios = {np.unique(query.fetch('mouse_id'))[0]: query.fetch('place_cell_ratio') * 100 for query in queries}
[np.mean([pc_ratios[k][i] for k in pc_ratios.keys()]) for i in range(5)]
[np.std([pc_ratios[k][i] for k in pc_ratios.keys() if k != 38]) for i in range(5)]

# Place cells/non place cells
place_cells = len(hheise_placecell.PlaceCell.ROI & pks & 'is_place_cell=1')
non_place_cells = len(common_img.Segmentation.ROI & pks & 'accepted=1') - place_cells



#%% Figure 2

# Microsphere behavior plot
norm_performance = (hheise_behav.VRPerformance & f'mouse_id in {helper.in_query(mouse_ids)}').get_normalized_performance(baseline_days=3, plotting=False)
prism_performance = norm_performance.pivot(index='day', columns='mouse_id', values='performance')
prism_performance.to_csv(os.path.join(folder, 'figure2', 'microsphere_behavior_groups.csv'))

# Sphere/lesion count vs performance group
spheres = pd.DataFrame((hheise_hist.MicrosphereSummary.Metric & 'metric_name="spheres"' & f'mouse_id in {helper.in_query(mouse_ids)}').fetch('mouse_id', 'count_extrap', as_dict=True))
spheres = spheres.rename(columns=dict(count_extrap='spheres'))
lesions = pd.DataFrame((hheise_hist.MicrosphereSummary.Metric & 'metric_name="auto"' & f'mouse_id in {helper.in_query(mouse_ids)}').fetch('mouse_id', 'count_extrap', as_dict=True))
lesions = lesions.rename(columns=dict(count_extrap='lesion'))
histo = pd.merge(spheres, lesions, on='mouse_id', how='outer').merge(grouping_2).sort_values(by='group')
histo.to_csv(os.path.join(folder, 'figure2', 'histo_summary.csv'))

# Running speed across time
data = []
for mouse in mouse_ids:
    day = (hheise_behav.VRPerformance & f'mouse_id = {mouse}').fetch('day')
    speed = (hheise_behav.VRPerformance & f'mouse_id = {mouse}').get_mean('mean_speed')
    running_speed = (hheise_behav.VRPerformance & f'mouse_id = {mouse}').get_mean('mean_running_speed')
    trial_duration = (hheise_behav.VRPerformance & f'mouse_id = {mouse}').get_mean('trial_duration')

    surg_day = (common_mice.Surgery & 'surgery_type="Microsphere injection"' &
                f'mouse_id = {mouse}').fetch1('surgery_date').date()
    day_diff = day-surg_day
    rel_day = [d.days for d in day_diff]

    data.append(pd.DataFrame(dict(mouse_id=mouse, date=day, rel_day=rel_day, speed=speed, running_speed=running_speed,
                                  trial_duration=trial_duration)))
data = pd.concat(data, ignore_index=True)
data = data.merge(grouping_2)

# fig, ax = plt.subplots(3, 1)
# sns.lineplot(data=data, x='rel_day', y='speed', hue='mouse_id', ax=ax[0])
# sns.lineplot(data=data, x='rel_day', y='running_speed', hue='mouse_id', ax=ax[1])
# sns.lineplot(data=data, x='rel_day', y='trial_duration', hue='mouse_id', ax=ax[2])
#
fig, ax = plt.subplots(3, 1)
sns.lineplot(data=data, x='rel_day', y='speed', hue='group', ax=ax[0])
sns.lineplot(data=data, x='rel_day', y='running_speed', hue='group', ax=ax[1])
sns.lineplot(data=data, x='rel_day', y='trial_duration', hue='group', ax=ax[2])

sns.lineplot(data=data, x='rel_day', y='speed', hue='group')
sns.lineplot(data=data, x='rel_day', y='running_speed', hue='group')

data_exp = data.pivot(index='rel_day', columns='mouse_id', values='running_speed')
data_exp.to_csv(os.path.join(folder, 'figure2', 'running_speed.csv'))

#%% Figure 3
# Preliminary post-stroke analysis of deficit mice (M41 and M121)

# Queries including post-stroke period
queries = ((common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=41'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=69' & 'day<="2021-03-23"'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=121' & 'day<"2022-09-09"'))

queries = ((common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=115' & 'day<"2022-09-09"'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=122' & 'day<"2022-09-09"'))

queries = ((common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=41'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=69' & 'day<="2021-03-23"'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=121' & 'day<"2022-09-09"'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=115' & 'day<"2022-09-09"'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=122' & 'day<"2022-09-09"')
           )

is_pc = []
pfs = []
spatial_maps = []
match_matrices = []
spat_dff_maps = []

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
                                               return_array=True, relative_dates=True,
                                               surgery='Microsphere injection'))

    spat_dff_maps.append(query.get_matched_data(table=hheise_placecell.BinnedActivity.ROI, attribute='bin_activity',
                                                extra_restriction=dict(corridor_type=0, place_cell_id=2),
                                                return_array=True, relative_dates=True,
                                                surgery='Microsphere injection'))


def spatial_map_correlations_single_cells(spatial_maps=list):
    """
    Correlate spatial maps across sessions of multiple numpy arrays. This function does not require the matching
    and merging of arrays and is more flexible with variable acquisition days.

    Args:
        spatial_maps: Data list, output from get_matched_data().

    Returns:
        A Dataframe with all the cross-session correlations.
    """

    DAY_DIFF = 3    # The day difference between sessions to be compared (usually 3)

    df_list = []
    for animal in spatial_maps:
        for net_id, data in animal.items():
            days = np.array(data[1])
            arr = data[0]

            curr_df = pd.DataFrame({'net_id': [net_id]*len(arr)})

            # Loop through days and compute correlation between sessions that are 3 days apart
            for day_idx, day in enumerate(days):
                next_day_idx = np.where(days == day+DAY_DIFF)[0]

                # If a session 3 days later exists, compute the correlation of all cells between these sessions
                # Do not analyze session 1 day after stroke (unreliable data)
                if day+DAY_DIFF != 1 and len(next_day_idx) == 1:
                    curr_corr = [np.corrcoef(arr[cell_id, day_idx], arr[cell_id, next_day_idx[0]])[0, 1]
                                 for cell_id in range(len(arr))]

                    curr_df[days[next_day_idx[0]]] = curr_corr
            df_list.append(curr_df)

    final_df = pd.concat(df_list, ignore_index=True)

    # Sort columns numerically. The column names show the 2nd day of correlation, e.g. the column 0 shows the correlation
    # of activity on day 0 (the day of the surgery) with day -3 (3 days before surgery)
    net_ids = final_df.pop('net_id')
    sorted_days = np.sort(final_df.columns.astype(int))
    final_df = final_df[sorted_days]
    final_df['net_id'] = net_ids

    ### ANALYSIS ###
    # Get average cross-correlation of of pre, early post and late post sessions

    # Prestroke sessions (the index has to be the column names, not a boolean mask)
    pre_avg = np.tanh(np.nanmean(np.arctanh(final_df[sorted_days[sorted_days <= 0]]), axis=1))

    # Pre-Post
    pre_post_avg = np.tanh(np.nanmean(np.arctanh(final_df[sorted_days[(0 < sorted_days) & (sorted_days <= DAY_DIFF)]]),
                                      axis=1))

    # Early Post
    early_post_avg = np.tanh(np.nanmean(np.arctanh(final_df[sorted_days[(0 < sorted_days) & (sorted_days <= 9)]]),
                                        axis=1))

    # Late Post
    late_post_avg = np.tanh(np.nanmean(np.arctanh(final_df[sorted_days[(9 < sorted_days)]]), axis=1))

    # All Post
    all_post_avg = np.tanh(np.nanmean(np.arctanh(final_df[sorted_days[(0 < sorted_days)]]), axis=1))

    # Construct DataFrame
    avg_df = pd.DataFrame({'pre': pre_avg, 'pre_post': pre_post_avg, 'early_post': early_post_avg,
                           'late_post': late_post_avg, 'all_post': all_post_avg})

    avg_df_rel = avg_df.div(avg_df['pre'], axis=0)
    avg_df_dif = avg_df.sub(avg_df['pre'], axis=0)

    # Drop cells that are not in every period
    avg_df_clean = avg_df.dropna(axis=0)
    avg_df_clean.to_csv(os.path.join(folder, r'single_cell_corr_sham.csv'))

    # Sort cells by pre-stroke correlation
    avg_df_clean_sort = avg_df_clean.sort_values(by='pre', ascending=False)
    avg_df_clean_sort.to_csv(os.path.join(folder, r'single_cell_corr_sorted_allstroke.csv'))

    avg_df_dif.to_csv(os.path.join(folder, r'single_cell_corr_dif_allstroke.csv'))

    # Store data sorted by mice
    for net in final_df['net_id'].unique():
        out = avg_df_dif.loc[final_df['net_id'] == net]
        out.to_csv(os.path.join(folder, f'single_cell_corr_dif_{net}.csv'))

    # Store total correlation pair data sorted by mice (for scatter plot)
    avg_df_clean['net_id'] = final_df['net_id']
    avg_df_totalpost = avg_df_clean.pivot(index='pre', columns='net_id', values='all_post')
    avg_df_totalpost.to_csv(os.path.join(folder, f'single_cell_corr_totalpost_allmice.csv'))


def place_cell_stability_fractions_score(is_pc):
    """ Code for the raw output from get_matched_data(). """

    # Remapping fractions pre vs post stroke (same as in fig 1)
    dfs = []
    population = []

    BACKWARDS = True    # If true, a place cell's stability is measured by its activity 3 days before the current day instead of 3 days after
    DAY_DIFF = 3        # The day difference between sessions to be compared (usually 3)

    for idx, animal in enumerate(is_pc):
        for net_id, data in animal.items():

            days = np.array(data[1])
            arr = data[0]

            # Loop through days and compute correlation between sessions that are 3 days apart
            for day_idx, day in enumerate(days):
                next_day_idx = np.where(days == day + DAY_DIFF)[0]

                # If a session 3 days later exists, compute the correlation of all cells between these sessions
                # Do not analyze session 1 day after stroke (unreliable data)
                if day + DAY_DIFF != 1 and len(next_day_idx) == 1:

                    if BACKWARDS:
                        i = next_day_idx[0]
                        i_next = day_idx
                    else:
                        i_next = next_day_idx[0]
                        i = day_idx

                    n_pcs = np.nansum(arr[:, i])
                    remaining_pc_idx = np.where(np.nansum(arr[:, [i, i_next]], axis=1) == 2)[0]
                    n_remaining_pcs = len(remaining_pc_idx)
                    # Get Place Fields of the same mouse/network/cells. Convert to array for indexing, then to list for accessibility
                    pfs_1 = list(np.array(pfs[idx][net_id][0])[remaining_pc_idx, i])
                    pfs_2 = list(np.array(pfs[idx][net_id][0])[remaining_pc_idx, i_next])

                    # For each stable place cell, compare place fields
                    for cell_idx, pf_1, pf_2 in zip(remaining_pc_idx, pfs_1, pfs_2):

                        # Get center of mass for current place fields in both sessions
                        pf_com_1 = [dc.place_field_com(spatial_map_data=spat_dff_maps[idx][net_id][0][cell_idx, i],
                                                       pf_indices=pf) for pf in pf_1]
                        pf_com_2 = [dc.place_field_com(spatial_map_data=spat_dff_maps[idx][net_id][0][cell_idx, i_next],
                                                       pf_indices=pf) for pf in pf_2]

                        # For each place field in day i, check if there is a place field on day 2 that overlaps with its CoM
                        pc_is_stable = np.nan
                        dist = np.nan
                        if len(pf_com_1) == len(pf_com_2):
                            # If cell has one PF on both days, check if the day-1 CoM is located inside day-2 PF -> stable
                            if len(pf_com_1) == 1:
                                if pf_2[0][0] < pf_com_1[0][0] < pf_2[0][-1]:
                                    pc_is_stable = True
                                else:
                                    pc_is_stable = False

                                # Compute distance for now only between singular PFs (positive distance means shift towards end)
                                dist = pf_com_2[0][0] - pf_com_1[0][0]

                            # If the cell has more than 1 PF on both days, check if all PFs overlap -> stable
                            else:
                                same_pfs = [True if pf2[0] < pf_com1[0] < pf2[-1] else False for pf_com1, pf2 in
                                            zip(pf_com_1, pf_2)]
                                pc_is_stable = all(same_pfs)  # This sets pc_is_stable to True if all PFs match, otherwise its False

                        # If cell has different number of place fields on both days, for now consider as unstable
                        else:
                            pc_is_stable = False

                        # Get difference in place field numbers to quantify how many cells change number of place fields
                        pf_num_change = len(pf_com_2) - len(pf_com_1)

                        dfs.append(pd.DataFrame([dict(net_id=net_id, cell_idx=cell_idx,
                                                      day=days[i], other_day=days[i_next],
                                                      stable=pc_is_stable, dist=dist, num_pf_1=len(pf_com_1),
                                                      num_pf_2=len(pf_com_2), pf_num_change=pf_num_change)]))

                    population.append(
                        pd.DataFrame([dict(net_id=net_id, day=days[i], other_day=days[i_next],
                                           n_pc=n_pcs, n_remaining=n_remaining_pcs,
                                           frac=n_remaining_pcs / n_pcs)]))

    population_df = pd.concat(population)
    stability_df = pd.concat(dfs)

    # Summarize stability by days
    stab_summ = []
    for net in np.sort(stability_df['net_id'].unique()):
        curr_net = stability_df[stability_df['net_id'] == net]

        for day in np.sort(curr_net['day'].unique()):
            curr_day = curr_net[curr_net['day'] == day]

            stab_summ.append(pd.DataFrame([dict(net_id=net, day=day, n_cells=len(curr_day),
                                                n_stable=curr_day['stable'].sum(),
                                                frac_stable=curr_day['stable'].sum() / len(curr_day))]))
    stab_summ = pd.concat(stab_summ)

    # Group by phases
    stab_summ_phase = []
    for net in np.sort(stab_summ['net_id'].unique()):

        curr_net = stab_summ[stab_summ['net_id'] == net]
        first_post_day = curr_net['day'].unique()[np.searchsorted(curr_net['day'].unique(), 0, side='right')]
        stab_summ_phase.append(pd.DataFrame([dict(net_id=net,
                                                  pre=curr_net[curr_net['day'] <= 0]['frac_stable'].mean(),
                                                  pre_post=curr_net[curr_net['day'] == first_post_day]['frac_stable'][0],
                                                  early=curr_net[(curr_net['day'] > first_post_day) & (curr_net['day'] <= 9)]['frac_stable'].mean(),
                                                  late=curr_net[curr_net['day'] > 9]['frac_stable'].mean())]))
    stab_summ_phase = pd.concat(stab_summ_phase)


    # Additionally, compute remainder and stability scores per cell (how often a cell is a (stable) PC in pre vs post)
    rem_scores = []
    for idx, animal in enumerate(is_pc):
        for net_id, data in animal.items():

            days = np.array(data[1])
            arr = data[0]

            remain_score = np.zeros((len(arr), 2)) * np.nan
            pre_mask = days <= 0
            for cell in range(len(arr)):
                if np.sum(~np.isnan(arr[cell, pre_mask])) > 2 and np.sum(~np.isnan(arr[cell, ~pre_mask])) > 3:
                    pre_score = np.nansum(arr[cell, pre_mask]) / np.nansum(~np.isnan(arr[cell, pre_mask]))
                    post_score = np.nansum(arr[cell, ~pre_mask]) / np.nansum(~np.isnan(arr[cell, ~pre_mask]))

                    remain_score[cell, 0] = pre_score
                    remain_score[cell, 1] = post_score

            rem_scores.append(remain_score)
    rem_score = np.vstack(rem_scores)

    # Reorganize stability_df to get cells in rows and days in columns (done together for all networks from stability_df)
    unique_days = np.sort(stability_df['day'].unique())
    if BACKWARDS:
        pre_stab_mask = unique_days <= 0
    else:
        pre_stab_mask = unique_days < 0

    df = pd.DataFrame(columns=unique_days)
    for idx, row in stability_df.iterrows():
        df.loc[row['cell_idx'], row['day']] = row['stable']
    stable_pc = np.asarray(df, dtype=float)

    stab_score = np.zeros((len(stable_pc), 2)) * np.nan
    for cell in range(len(stable_pc)):
        if np.sum(~np.isnan(stable_pc[cell, pre_stab_mask])) > 0 and np.sum(~np.isnan(stable_pc[cell, ~pre_stab_mask])) > 0:
            pre_score = np.nansum(stable_pc[cell, pre_stab_mask]) / np.nansum(~np.isnan(stable_pc[cell, pre_stab_mask]))
            post_score = np.nansum(stable_pc[cell, ~pre_stab_mask]) / np.nansum(~np.isnan(stable_pc[cell, ~pre_stab_mask]))

            stab_score[cell, 0] = pre_score
            stab_score[cell, 1] = post_score

    np.savetxt(os.path.join(folder, r'single_cell_remain_score.csv'), rem_score)
    np.savetxt(os.path.join(folder, r'single_cell_stab_score.csv'), stab_score)


def place_cell_stability_score(stability_df):
    """ Code for cleaned filtered DataFrame. """
    # Give each tracked place cell from the first poststroke session stability scores: how often is the cell a PC before
    # and after stroke?

    first_post_mask = placecell_data[:, 5] == 1
    first_post_pc = placecell_data[first_post_mask]

    remain_score = np.zeros((len(first_post_pc), 2)) * np.nan
    for cell in range(len(first_post_pc)):

        # Only use cells that have more than 2 prestroke or poststroke sessions tracked
        if np.sum(~np.isnan(first_post_pc[cell, :5])) > 2 and np.sum(~np.isnan(first_post_pc[cell, 6:])) > 2:
            pre_score = np.nansum(first_post_pc[cell, :5]) / np.nansum(~np.isnan(first_post_pc[cell, :5]))
            post_score = np.nansum(first_post_pc[cell, 6:]) / np.nansum(~np.isnan(first_post_pc[cell, 6:]))

            remain_score[cell, 0] = pre_score
            remain_score[cell, 1] = post_score

    # Reorganize stability_df to get cells in rows and days in columns
    df = pd.DataFrame(index=np.arange(len(placefield_data))[first_post_mask], columns=stability_df['day'].unique())
    for idx, row in stability_df.iterrows():
        df.loc[row['cell_idx'], row['day']] = row['stable']
    stable_pc = np.asarray(df, dtype=float)

    # Calculate average stability across prestroke and poststroke session pairs (excluding pre vs post)
    pre_score = np.nanmean(stable_pc[:, :2], axis=1)
    post_score = np.nanmean(stable_pc[:, 3:], axis=1)
    stab_score = np.stack([pre_score, post_score])



#%%
### Legacy functions that work only with filtered, date-matched DataFrames
def place_cell_population_stability():
    """ Code for a single, filtered dataframe. """
    # Remapping pre vs post stroke (same as in fig 1)
    dfs = []
    population = []

    # # Use this to examine the next session (forward)
    # for i in range(placecell_data.shape[1] - 1):
    #
    #     if i < 2:               # in prestroke, compare in 3-day intervals
    #         i_next = i + 3
    #     elif i in [2, 3]:       # only compare last prestroke session with first poststroke session
    #         continue
    #     else:
    #         i_next = i+1

    # Use this to examine the previous session (backwards)
    for i in range(3, placecell_data.shape[1]):

        if i < 5:  # in prestroke, compare in 3-day intervals
            i_next = i - 3
        else:
            i_next = i - 1

        # Idx of PCs in session i that are also PCs in session i+1
        n_pcs = np.nansum(placecell_data[:, i])
        stable_pc_idx = np.where(np.nansum(placecell_data[:, [i, i_next]], axis=1) == 2)[0]
        n_stable_pcs = len(stable_pc_idx)
        pfs_1 = placefield_data[stable_pc_idx, i]
        pfs_2 = placefield_data[stable_pc_idx, i_next]

        # For each stable place cell, compare place fields
        for cell_idx, pf_1, pf_2 in zip(stable_pc_idx, pfs_1, pfs_2):

            # Get center of mass for current place fields in both sessions
            pf_com_1 = [dc.place_field_com(spatial_map_data=spatial_dff_data[cell_idx, i], pf_indices=pf) for pf in pf_1]
            pf_com_2 = [dc.place_field_com(spatial_map_data=spatial_dff_data[cell_idx, i_next], pf_indices=pf) for pf in
                        pf_2]

            # For each place field in day i, check if there is a place field on day 2 that overlaps with its CoM
            pc_is_stable = np.nan
            dist = np.nan
            if len(pf_com_1) == len(pf_com_2):
                # If cell has one PF on both days, check if the day-1 CoM is located inside day-2 PF -> stable
                if len(pf_com_1) == 1:
                    if pf_2[0][0] < pf_com_1[0][0] < pf_2[0][-1]:
                        pc_is_stable = True
                    else:
                        pc_is_stable = False

                    # Compute distance for now only between singular PFs (positive distance means shift towards end)
                    dist = pf_com_2[0][0] - pf_com_1[0][0]

                # If the cell has more than 1 PF on both days, check if all PFs overlap -> stable
                else:
                    same_pfs = [True if pf2[0] < pf_com1[0] < pf2[-1] else False for pf_com1, pf2 in
                                zip(pf_com_1, pf_2)]
                    pc_is_stable = all(same_pfs)  # This sets pc_is_stable to True if all PFs match, otherwise its False

            # If cell has different number of place fields on both days, for now consider as unstable
            else:
                pc_is_stable = False

            # Get difference in place field numbers to quantify how many cells change number of place fields
            pf_num_change = len(pf_com_2) - len(pf_com_1)

            dfs.append(pd.DataFrame([dict(mouse_id=mouse_idx[cell_idx], cell_idx=cell_idx, day=is_pc[1]['121_1'][1][i], next_day=is_pc[1]['121_1'][1][i_next],
                                          stable=pc_is_stable, dist=dist, num_pf_1=len(pf_com_1), num_pf_2=len(pf_com_2),
                                          pf_num_change=pf_num_change)]))

        population.append(pd.DataFrame([dict(day=is_pc[1]['121_1'][1][i], next_day=is_pc[1]['121_1'][1][i_next], n_pc=n_pcs, n_remaining=n_stable_pcs,
                                             frac=n_stable_pcs/n_pcs)]))

    population_df = pd.concat(population)
    stability_df = pd.concat(dfs)

    # Summarize stability by days
    stab_summ = []
    for day in stability_df['day'].unique():
        curr_day = stability_df[stability_df['day'] == day]

        stab_summ.append(pd.DataFrame([dict(day=day, n_cells=len(curr_day), n_stable=curr_day['stable'].sum(),
                                            frac_stable=curr_day['stable'].sum()/len(curr_day))]))
    stab_summ = pd.concat(stab_summ)


# Compute spatial map correlations, legacy code for easy filtered dataframe
def spatial_map_correlations():
    """ Correlate spatial maps across sessions of a single DataFrame. """
    pearson_df = []
    for cell_id in range(map_data.shape[0]):

        # For each neuron, compute the correlation between all prestroke sessions 3 days apart (d1-d4, d2-d5)
        pre = [np.corrcoef(map_data[cell_id, sess1], map_data[cell_id, sess2])[0, 1] for sess1, sess2 in [(0, 3), (1, 4)]]
        pre_avg = np.tanh(np.nanmean(np.arctanh(pre)))

        # Pre vs Post (last prestroke day vs first poststroke day)
        pre_post = np.corrcoef(map_data[cell_id, 4], map_data[cell_id, 5])[0, 1]

        # All poststroke session combinations (are by default 3 days apart) and compute averages
        post = [np.corrcoef(map_data[cell_id, i], map_data[cell_id, i+1])[0, 1] for i in range(5, map_data.shape[1]-1)]
        early_avg = np.tanh(np.nanmean(np.arctanh(post[:3])))
        late_avg = np.tanh(np.nanmean(np.arctanh(post[3:])))
        total_avg = np.tanh(np.nanmean(np.arctanh(post)))

        # Add data to dataframe
        pearson_df.append(pd.DataFrame(data=[dict(mouse=mouse_idx[cell_id], cell_id=cell_id,
                                                  pre=pre_avg, pre_post=pre_post, early_post=early_avg, late_post=late_avg, total_post=total_avg,
                                                  pre_post_rel=pre_post/pre_avg, early_post_rel=early_avg/pre_avg, late_post_rel=late_avg/pre_avg, total_post_rel=total_avg/pre_avg,
                                                  pre_post_dif=pre_post-pre_avg, early_post_dif=early_avg-pre_avg, late_post_dif=late_avg-pre_avg, total_post_dif=total_avg-pre_avg)]))

    pearson_df = pd.concat(pearson_df, ignore_index=True)

    # Drop rows with NA
    pearson_df_clean = pearson_df.dropna(axis=0)

    # Export for prism
    raw = pearson_df_clean.iloc[:, 2:7].T.to_csv(os.path.join(folder, r'single_cell_corr_raw.csv'))
    rel = pearson_df_clean.iloc[:, 7:11].T.to_csv(os.path.join(folder, r'single_cell_corr_rel.csv'))
    dif = pearson_df_clean.iloc[:, 11:].T.to_csv(os.path.join(folder, r'single_cell_corr_dif.csv'))


def place_cell_single_stability(stability_df):
    """ Code for cleaned filtered DataFrame. """
    # Give each tracked place cell from the first poststroke session stability scores: how often is the cell a PC before
    # and after stroke?

    first_post_mask = placecell_data[:, 5] == 1
    first_post_pc = placecell_data[first_post_mask]

    remain_score = np.zeros((len(first_post_pc), 2)) * np.nan
    for cell in range(len(first_post_pc)):

        # Only use cells that have more than 2 prestroke or poststroke sessions tracked
        if np.sum(~np.isnan(first_post_pc[cell, :5])) > 2 and np.sum(~np.isnan(first_post_pc[cell, 6:])) > 2:
            pre_score = np.nansum(first_post_pc[cell, :5]) / np.nansum(~np.isnan(first_post_pc[cell, :5]))
            post_score = np.nansum(first_post_pc[cell, 6:]) / np.nansum(~np.isnan(first_post_pc[cell, 6:]))

            remain_score[cell, 0] = pre_score
            remain_score[cell, 1] = post_score

    # Reorganize stability_df to get cells in rows and days in columns
    df = pd.DataFrame(index=np.arange(len(placefield_data))[first_post_mask], columns=stability_df['day'].unique())
    for idx, row in stability_df.iterrows():
        df.loc[row['cell_idx'], row['day']] = row['stable']
    stable_pc = np.asarray(df, dtype=float)

    # Calculate average stability across prestroke and poststroke session pairs (excluding pre vs post)
    pre_score = np.nanmean(stable_pc[:, :2], axis=1)
    post_score = np.nanmean(stable_pc[:, 3:], axis=1)
    stab_score = np.stack([pre_score, post_score])
