#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 11/05/2023 12:53
@author: hheise

"""

import numpy as np
import pandas as pd
from skimage import measure
import matplotlib
import matplotlib.pyplot as plt
from scipy import spatial
import bisect

from schema import common_match, common_img, hheise_placecell
from util import helper

from preprint import data_cleaning as dc

DAY_DIFF = 3
BACKWARDS = True

# Deficit
queries = ((common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=41'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=69' & 'day<="2021-03-23"'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=121' & 'day<"2022-09-09"'))

# Sham
queries = ((common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=115' & 'day<"2022-09-09"'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=122' & 'day<"2022-09-09"'))

# All
queries = ((common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=41'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=69' & 'day<="2021-03-23"'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=121' & 'day<"2022-09-09"'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=115' & 'day<"2022-09-09"'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=122' & 'day<"2022-09-09"')
           )

spatial_maps = []
match_matrices = []

for query in queries:
    match_matrices.append(query.construct_matrix())

    # is_pc.append(query.get_matched_data(table=hheise_placecell.PlaceCell.ROI, attribute='is_place_cell',
    #                                     extra_restriction=dict(corridor_type=0, place_cell_id=2),
    #                                     return_array=True, relative_dates=True, surgery='Microsphere injection'))
    #
    # pfs.append(query.get_matched_data(table=hheise_placecell.PlaceCell.PlaceField, attribute='bin_idx',
    #                                   extra_restriction=dict(corridor_type=0, place_cell_id=2),
    #                                   return_array=False, relative_dates=True, surgery='Microsphere injection'))

    spatial_maps.append(query.get_matched_data(table=hheise_placecell.BinnedActivity.ROI, attribute='bin_spikerate',
                                               extra_restriction=dict(corridor_type=0, place_cell_id=2),
                                               return_array=True, relative_dates=True,
                                               surgery='Microsphere injection'))

    # spat_dff_maps.append(query.get_matched_data(table=hheise_placecell.BinnedActivity.ROI, attribute='bin_activity',
    #                                             extra_restriction=dict(corridor_type=0, place_cell_id=2),
    #                                             return_array=True, relative_dates=True,
    #                                             surgery='Microsphere injection'))


#%% Compute cross-session stability

def compute_crosssession_stability(spat_maps, days):
    """ Correlate spatial maps across days for a single network. """


    curr_df = pd.DataFrame({})

    # Loop through days and compute correlation between sessions that are 3 days apart
    for day_idx, day in enumerate(days):
        next_day_idx = np.where(days == day+DAY_DIFF)[0]

        # If a session 3 days later exists, compute the correlation of all cells between these sessions
        # Do not analyze session 1 day after stroke (unreliable data)
        if day+DAY_DIFF != 1 and len(next_day_idx) == 1:
            curr_corr = [np.corrcoef(spat_maps[cell_id, day_idx], spat_maps[cell_id, next_day_idx[0]])[0, 1]
                         for cell_id in range(len(spat_maps))]

            curr_df[days[next_day_idx[0]]] = curr_corr

    # Just to make sure, sort days
    sorted_days = np.sort(curr_df.columns.astype(int))
    curr_df = curr_df[sorted_days]

    return curr_df


def group_sessions(df, drop_na=True):

    days = df.columns.astype(int)

    pre_avg = np.tanh(np.nanmean(np.arctanh(df[days[days <= 0]]), axis=1))

    # Pre-Post
    pre_post_avg = np.tanh(np.nanmean(np.arctanh(df[days[(0 < days) & (days <= DAY_DIFF)]]), axis=1))
    # Early Post
    early_post_avg = np.tanh(np.nanmean(np.arctanh(df[days[(0 < days) & (days <= 9)]]), axis=1))
    # Late Post
    late_post_avg = np.tanh(np.nanmean(np.arctanh(df[days[(9 < days)]]), axis=1))
    # All Post
    all_post_avg = np.tanh(np.nanmean(np.arctanh(df[days[(0 < days)]]), axis=1))

    # Construct DataFrame
    avg_df = pd.DataFrame({'pre': pre_avg, 'pre_post': pre_post_avg, 'early_post': early_post_avg,
                           'late_post': late_post_avg, 'all_post': all_post_avg})

    if drop_na:
        avg_df = avg_df.dropna(axis=0)

    return avg_df


def get_neighbourhood_avg(match_matrix, mouse_id, day, values, neighborhood_radius=50, n_shuffle=1000):

    # Fetch CoMs of all neurons and create distance tree
    username = 'hheise'
    key = dict(username=username, mouse_id=mouse_id, day=day)
    # Get the global mask_id for each row_id/matched cell in the match_matrix
    mask_ids = np.array(match_matrix[[col for col in match_matrix.columns if day in col][0]].iloc[values.index], dtype=int)
    coms = (common_img.Segmentation.ROI & key & f'mask_id in {helper.in_query(mask_ids)}').fetch('com')
    coms_list = [list(com) for com in coms]
    neighbor_tree = spatial.KDTree(coms_list)  # Build kd-tree to query nearest neighbours of any ROI

    df = []
    for rel_idx in range(len(coms)):

        distance, index = neighbor_tree.query(coms_list[rel_idx], k=len(coms_list))

        # get the number of ROIs in the neighbourhood and their relative indices
        num_neurons_in_radius = bisect.bisect(distance, neighborhood_radius) - 1  # -1 to not count the ROI itself
        index_in_radius = index[1: max(0, num_neurons_in_radius) + 1]     # start at 1 to not count the ROI itself

        neighbourhood_mean = values.iloc[index_in_radius].mean()

        if num_neurons_in_radius > 0:
            # Shuffle correlation values to get the average neighbour correlation if there was no relation to distance
            shuffle = [np.random.choice(values, size=num_neurons_in_radius, replace=False).mean() for n in range(n_shuffle)]
            # The p-value is the probability to have a higher mean neighbourhood correlation just by random sampling
            shuffle_p = np.sum(np.array(shuffle) > neighbourhood_mean)/n_shuffle
        else:
            shuffle_p = np.nan

        df.append(pd.DataFrame(data=[dict(idx=values.index[rel_idx], value=values.iloc[rel_idx],
                                          num_neighbours=num_neurons_in_radius,
                                          neighbour_mean=neighbourhood_mean, p=shuffle_p)]))
    df = pd.concat(df)

    return


def compute_placecell_stability(spat_dff_maps, place_fields, days):

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
            pfs_1 = list(np.array(place_fields[idx][net_id][0])[remaining_pc_idx, i])
            pfs_2 = list(np.array(place_fields[idx][net_id][0])[remaining_pc_idx, i_next])

            # For each stable place cell, compare place fields
            for cell_idx, pf_1, pf_2 in zip(remaining_pc_idx, pfs_1, pfs_2):

                # Get center of mass for current place fields in both sessions
                pf_com_1 = [dc.place_field_com(spatial_map_data=spat_dff_maps[cell_idx, i],
                                               pf_indices=pf) for pf in pf_1]
                pf_com_2 = [dc.place_field_com(spatial_map_data=spat_dff_maps[cell_idx, i_next],
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
                        pc_is_stable = all(
                            same_pfs)  # This sets pc_is_stable to True if all PFs match, otherwise its False

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


def plot_colored_cells(match_matrix, mouse_id, day, row_ids, color, cmap='magma', contour_thresh=0.05,
                       background='avg_image', axis=None, title=None):

    # Fetch background image and initialize colormap
    username = 'hheise'
    key = dict(username=username, mouse_id=mouse_id, day=day)
    bg = (common_img.QualityControl & key).fetch1(background)
    colormap = matplotlib.cm.get_cmap(cmap)
    norm = matplotlib.colors.Normalize(vmin=np.min(color), vmax=np.max(color))

    # Get the global mask_id for each row_id/matched cell in the match_matrix
    mask_ids = np.array(match_matrix[[col for col in match_matrix.columns if day in col][0]].iloc[row_ids], dtype=int)

    # Get contours for all cells
    footprints = (common_img.Segmentation.ROI & key & f'mask_id in {helper.in_query(mask_ids)}').get_rois()
    contours = [measure.find_contours(footprint, contour_thresh, fully_connected='high')[0] for footprint in footprints]

    # Draw the background image
    if axis is None:
        axis = plt.figure().gca()
    axis.imshow(bg, cmap='gray', vmin=np.percentile(bg, 5), vmax=np.percentile(bg, 95))

    # Draw each contour as a filled polygon, taking the color from the given colormap
    for c, col in zip(contours, color):

        c[:, [0, 1]] = c[:, [1, 0]]  # Plotting swaps X and Y axes, swap them back before
        # ax.plot(*c.T, c='black')
        axis.add_patch(matplotlib.patches.Polygon(c, facecolor=colormap(norm(col)), alpha=0.8))

    # Draw colorbar for patch color
    plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axis)

    if title is not None:
        axis.set_title(title)

    return axis


# Compute cross-session stability for each network
data = {41: compute_crosssession_stability(spatial_maps[0]['41_1'][0], np.array(spatial_maps[0]['41_1'][1])),
        69: compute_crosssession_stability(spatial_maps[1]['69_1'][0], np.array(spatial_maps[1]['69_1'][1])),
        121: compute_crosssession_stability(spatial_maps[2]['121_1'][0], np.array(spatial_maps[2]['121_1'][1])),
        115: compute_crosssession_stability(spatial_maps[3]['115_1'][0], np.array(spatial_maps[3]['115_1'][1])),
        122: compute_crosssession_stability(spatial_maps[4]['122_1'][0], np.array(spatial_maps[4]['122_1'][1]))}

# Average correlations across time periods
avg_data = {k: group_sessions(v) for k, v in data.items()}

# Compute absolute difference in correlation compared to pre-stroke
avg_dif = {k: v.sub(v['pre'], axis=0) for k, v in avg_data.items()}


## Plot cells in the FOV, shaded by the correlation change in early poststroke compared to prestroke
fig, ax = plt.subplots(2, 3)

ax[0, 0] = plot_colored_cells(match_matrix=match_matrices[0]['41_1'], mouse_id=41, day='2020-08-27', axis=ax[0, 0],
                              row_ids=avg_dif[41].index, color=avg_dif[41]['early_post'], title='41')
ax[0, 1] = plot_colored_cells(match_matrix=match_matrices[1]['69_1'], mouse_id=69, day='2021-03-11', axis=ax[0, 1],
                              row_ids=avg_dif[69].index, color=avg_dif[69]['early_post'], title='69')
ax[0, 2].set_visible(False)
ax[1, 0] = plot_colored_cells(match_matrix=match_matrices[2]['121_1'], mouse_id=121, day='2022-08-15', axis=ax[1, 0],
                              row_ids=avg_dif[121].index, color=avg_dif[121]['early_post'], title='121')
ax[1, 1] = plot_colored_cells(match_matrix=match_matrices[3]['115_1'], mouse_id=115, day='2022-08-12', axis=ax[1, 1],
                              row_ids=avg_dif[115].index, color=avg_dif[115]['early_post'], title='115')
ax[1, 2] = plot_colored_cells(match_matrix=match_matrices[4]['122_1'], mouse_id=122, day='2022-08-15', axis=ax[1, 2],
                              row_ids=avg_dif[122].index, color=avg_dif[122]['early_post'], title='122')

# Todo: Compute average stability of neighbouring cells (e.g. 100 px neighbourhood) of each cell, and correlate against
#  stability of center cell. Maybe check if stability around stable cells is higher than for shuffled cells?

## Split cells into stable_pre-stable_post and unstable_pre-stable_post
for k, v in avg_data.items():

    pre_stable = v['pre'] >= np.percentile(v['pre'], 75)
    pre_unstable = v['pre'] <= np.percentile(v['pre'], 25)
    post_stable = v['early_post'] >= np.percentile(v['early_post'], 75)

    remain_stable = pre_stable & post_stable
    became_stable = pre_unstable & post_stable






avg_df_rel = avg_df.div(avg_df['pre'], axis=0)
avg_df_dif = avg_df.sub(avg_df['pre'], axis=0)

# Drop cells that are not in every period
avg_df_clean = avg_df.dropna(axis=0)

# Sort cells by pre-stroke correlation
avg_df_clean_sort = avg_df_clean.sort_values(by='pre', ascending=False)

# Store data sorted by mice
for net in final_df['net_id'].unique():
    out = avg_df_dif.loc[final_df['net_id'] == net]

# Store total correlation pair data sorted by mice (for scatter plot)
avg_df_clean['net_id'] = final_df['net_id']
avg_df_totalpost = avg_df_clean.pivot(index='pre', columns='net_id', values='all_post')


