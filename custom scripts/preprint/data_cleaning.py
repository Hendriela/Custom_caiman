#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 04/04/2023 16:48
@author: hheise

Function that handle and clean data, especially single-cell data, for further analysis
"""
import numpy as np
from typing import Tuple
import pickle
import os
import pandas as pd

from schema import hheise_placecell, common_img


def filter_matched_data(match_list: list, keep_nomatch=False):
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

    try:
        data_array = []
        mouse_id_list = []
        for curr_data in match_list:
            for key, net in curr_data.items():
                data_array.append(net[0])
                mouse_id_list.extend([int(key.split('_')[0])] * net[0].shape[0])
        data_array = np.vstack(data_array)
        mouse_id_list = np.array(mouse_id_list)
    except ValueError:
        print('Data arrays have different dimensions. Merging as DataFrame instead')
        dfs = []
        for curr_data in match_list:
            for key, net in curr_data.items():
                dfs.append(pd.DataFrame(net[0], columns=net[1]))
                dfs[-1]['net_id'] = key

        df = pd.concat(dfs)

        # Sort columns by date value
        net_ids = df.pop('net_id')
        cols = np.sort(df.columns.astype(int))
        df = df[cols]
        df['net_id'] = net_ids
        d1 = df.pop(1)  # Day 1 should not be analyzed

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


def place_field_com(spatial_map_data, pf_indices) -> Tuple[float, float]:
    """
    Compute center of mass for place fields of a single cell.

    Args:
        spatial_map_data: 1D array, spatial activity map of one cell
        pf_indices: Array holding indices of one place field

    Returns:
        Center-of-mass of the current place field based on the whole spatial map, and its standard deviation
    """

    spat_map = spatial_map_data[pf_indices]

    # Normalize
    map_norm = (spat_map - np.min(spat_map)) / (np.max(spat_map) - np.min(spat_map))
    # Convert to Probability Mass Function / Probability distribution
    map_pmf = map_norm / np.sum(map_norm)
    # Calculate moment (center of mass)
    com = float(np.sum(np.arange(len(map_pmf)) * map_pmf))

    # Calculate standard deviation
    com_std = []
    for t in np.arange(len(map_pmf)):
        com_std.append((t ** 2 * map_pmf[t]) - com ** 2)
    com_std = float(np.sqrt(np.sum(np.arange(len(map_pmf)) ** 2 * map_pmf) - com ** 2))

    # plt.plot(spat_map)
    # plt.axvline(com, color='r')
    # plt.axvspan(com - com_std / 2, com + com_std / 2, color='r', alpha=0.3)

    # Correct the center of mass to return map-wide indices before returning
    return com + pf_indices[0], com_std


def sync_days(match_list):
    dates = {}
    for list_idx, curr_data in enumerate(match_list):
        for key, net in curr_data.items():
            # Assume that...
            # ...the last 5 negative days are prestroke and will be treated as
            stroke_day_idx = np.where(np.array(net[1]) > 0)[0][0]
            synced_days = net[1][stroke_day_idx - 5]


def screen_place_cells():
    """ Screen place cells by showing spatial activity of all trials, to select good place cells. """

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


def load_data(data_type,
              folder: str=r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Data\analysis\matched_data'):

    match data_type:
        case data_type if data_type in ['spatial_activity_maps', 'dff_maps', 'spatial_act_maps', 'spat_act_maps',
                                        'spatial_activity_maps_dff', 'spat_dff_maps']:
            fname = 'spatial_activity_maps_dff.pkl'
        case data_type if data_type in ['spatial_decon_maps', 'decon_maps', 'spatial_maps', 'spat_maps',
                                        'spatial_activity_maps_spikerate']:
            fname = 'spatial_activity_maps_spikerate.pkl'
        case data_type if data_type in ['match_matrix', 'match_matrices', 'matrices']:
            fname = 'match_matrices.pkl'
        case data_type if data_type in ['is_pc', 'is_place_cell']:
            fname = 'is_pc.pkl'
        case data_type if data_type in ['pfs', 'place_fields', 'place_field_idx']:
            fname = 'pf_idx.pkl'

    with open(os.path.join(folder, fname), 'rb') as file:
        data = pickle.load(file)

    return data


def fix_place_cell_ratio():

    ### No mismatch for SpatialInformation place_cell_ratio
    pks, pcr = hheise_placecell.PlaceCell().fetch('KEY', 'place_cell_ratio')

    for key, ratio in zip(pks, pcr):
        num_pcs = len((hheise_placecell.PlaceCell.ROI & key & 'is_place_cell=1'))
        num_cells = len((common_img.Segmentation.ROI & key & 'accepted=1'))
        if np.float16(num_pcs/num_cells) != np.float16(ratio):
            print(f'Mismatch at {key}:\n{num_pcs/num_cells:.5f} vs PCR = {np.float16(ratio):.5f}')

            hheise_placecell.PlaceCell().update1(dict(**key, place_cell_ratio=num_pcs/num_cells))

    return
