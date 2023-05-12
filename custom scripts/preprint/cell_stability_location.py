#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 11/05/2023 12:53
@author: hheise

"""

import numpy as np
import pandas as pd
import seaborn as sns
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

is_pc = []
pfs = []
spatial_maps = []
match_matrices = []
spat_dff_maps = []

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

    return df


def compute_placecell_fractions(pc, days):

    # Get number of PC sessions in pre, early and late post for each cell
    pc_frac = pd.DataFrame(dict(pre_avg=np.nanmean(pc[:, days <= 0], axis=1),
                                early_post_avg=np.nanmean(pc[:, (0 < days) & (days <= 9)], axis=1),
                                late_post_avg=np.nanmean(pc[:, 9 < days], axis=1),
                                all_post_avg=np.nanmean(pc[:, 0 < days], axis=1)))

    # Classify cells into stable/unstable
    pc_class = np.zeros(len(pc), dtype=int)    # Class 0: Unclassified cells (never PCs)
    pc_class[(pc_frac['pre_avg'] == 0) & (pc_frac['early_post_avg'] > 0)] = 1  # Class 1: no PC pre, PC early post
    pc_class[(pc_frac['pre_avg'] > 0) & (pc_frac['early_post_avg'] == 0)] = 2  # Class 2: PC pre, no PC early post
    pc_class[(pc_frac['pre_avg'] >= 0.5) & (pc_frac['early_post_avg'] >= 0.5)] = 3  # Class 3: PC pre and PC early post

    return pc_class



def plot_colored_cells(match_matrix, mouse_id, day, row_ids, color, cmap='magma', contour_thresh=0.05, draw_cbar=True,
                       background='avg_image', axis=None, title=None, cbar_ticklabels=None):

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

    if draw_cbar:
        # Draw colorbar for patch color
        cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axis)

        if cbar_ticklabels:
            cbar.set_ticks([np.max(color) / len(cbar_ticklabels) * (i + 0.5) for i in range(len(cbar_ticklabels))])
            cbar.set_ticklabels(cbar_ticklabels)

    if title is not None:
        axis.set_title(title)

    return axis


data = {41: compute_crosssession_stability(spatial_maps[0]['41_1'][0], np.array(spatial_maps[0]['41_1'][1])),
        69: compute_crosssession_stability(spatial_maps[1]['69_1'][0], np.array(spatial_maps[1]['69_1'][1])),
        121: compute_crosssession_stability(spatial_maps[2]['121_1'][0], np.array(spatial_maps[2]['121_1'][1])),
        115: compute_crosssession_stability(spatial_maps[3]['115_1'][0], np.array(spatial_maps[3]['115_1'][1])),
        122: compute_crosssession_stability(spatial_maps[4]['122_1'][0], np.array(spatial_maps[4]['122_1'][1]))}

# Average correlations across time periods
avg_data = {k: group_sessions(v) for k, v in data.items()}

# Compute absolute difference in correlation compared to pre-stroke
avg_dif = {k: v.sub(v['pre'], axis=0) for k, v in avg_data.items()}

# Get neighbourhood average correlation diff for each neuron
avg_neighbour_dif = {41: get_neighbourhood_avg(match_matrix=match_matrices[0]['41_1'], mouse_id=41, day='2020-08-27',
                                               values=avg_dif[41]['early_post']),
                     69: get_neighbourhood_avg(match_matrix=match_matrices[1]['69_1'], mouse_id=69, day='2021-03-11',
                                               values=avg_dif[69]['early_post']),
                     121: get_neighbourhood_avg(match_matrix=match_matrices[2]['121_1'], mouse_id=121, day='2022-08-15',
                                                values=avg_dif[121]['early_post']),
                     115: get_neighbourhood_avg(match_matrix=match_matrices[3]['115_1'], mouse_id=115, day='2022-08-12',
                                                values=avg_dif[115]['early_post']),
                     122: get_neighbourhood_avg(match_matrix=match_matrices[4]['122_1'], mouse_id=122, day='2022-08-15',
                                                values=avg_dif[122]['early_post'])}


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


## Correlate correlation_diff and avg_neighbour_diff
colormap = matplotlib.cm.get_cmap('viridis')
fig, ax = plt.subplots(2, 3)
sns.regplot(data=avg_neighbour_dif[41], x='value', y='neighbour_mean', ax=ax[0, 0], scatter_kws={'color': colormap(avg_neighbour_dif[41]['p'][~np.isnan(avg_neighbour_dif[41]['p'])])})
sns.regplot(data=avg_neighbour_dif[69], x='value', y='neighbour_mean', ax=ax[0, 1], scatter_kws={'color': colormap(avg_neighbour_dif[69]['p'][~np.isnan(avg_neighbour_dif[69]['p'])])})
sns.regplot(data=avg_neighbour_dif[121], x='value', y='neighbour_mean', ax=ax[1, 0], scatter_kws={'color': colormap(avg_neighbour_dif[121]['p'][~np.isnan(avg_neighbour_dif[121]['p'])])})
sns.regplot(data=avg_neighbour_dif[115], x='value', y='neighbour_mean', ax=ax[1, 1], scatter_kws={'color': colormap(avg_neighbour_dif[115]['p'][~np.isnan(avg_neighbour_dif[115]['p'])])})
sns.regplot(data=avg_neighbour_dif[122], x='value', y='neighbour_mean', ax=ax[1, 2], scatter_kws={'color': colormap(avg_neighbour_dif[122]['p'][~np.isnan(avg_neighbour_dif[122]['p'])])})

# Todo: Plot colorbars for p-value --> investigate why pvalue tracks nicely with y-axis


## Split cells into stable_pre-stable_post and unstable_pre-stable_post
for k, v in avg_data.items():

    pre_stable = v['pre'] >= np.percentile(v['pre'], 75)
    pre_unstable = v['pre'] <= np.percentile(v['pre'], 25)
    post_stable = v['early_post'] >= np.percentile(v['early_post'], 75)

    remain_stable = pre_stable & post_stable
    became_stable = pre_unstable & post_stable


pc_fractions = {41: compute_placecell_fractions(pc=is_pc[0]['41_1'][0], days=np.array(is_pc[0]['41_1'][1])),
                69: compute_placecell_fractions(pc=is_pc[1]['69_1'][0], days=np.array(is_pc[1]['69_1'][1])),
                121: compute_placecell_fractions(pc=is_pc[2]['121_1'][0], days=np.array(is_pc[2]['121_1'][1])),
                115: compute_placecell_fractions(pc=is_pc[3]['115_1'][0], days=np.array(is_pc[3]['115_1'][1])),
                122: compute_placecell_fractions(pc=is_pc[4]['122_1'][0], days=np.array(is_pc[4]['122_1'][1]))}


# Make categorical colormap
cat_cm = matplotlib.colors.ListedColormap(np.array([[0.4, 0.2, 0.0, 1],     # Class 0: brown
                                                    [0.0, 0.6, 1.0, 1],     # Class 0: blue
                                                    [0.2, 0.8, 0.2, 1],     # Class 0: green
                                                    [1.0, 0.4, 0.0, 1]]))   # Class 0: orange

fig, ax = plt.subplots(2, 3)
ax[0, 0] = plot_colored_cells(match_matrix=match_matrices[0]['41_1'], mouse_id=41, day='2020-08-27', axis=ax[0, 0],
                              row_ids=np.arange(len(pc_fractions[41])), color=pc_fractions[41], title='41', cmap=cat_cm,
                              draw_cbar=False)
ax[0, 1] = plot_colored_cells(match_matrix=match_matrices[1]['69_1'], mouse_id=69, day='2021-03-11', axis=ax[0, 1],
                              row_ids=np.arange(len(pc_fractions[69])), color=pc_fractions[69], title='69', cmap=cat_cm,
                              cbar_ticklabels=['Never PC', 'PC post only', 'PC pre only', 'Always PC'])
ax[0, 2].set_visible(False)
ax[1, 0] = plot_colored_cells(match_matrix=match_matrices[2]['121_1'], mouse_id=121, day='2022-08-15', axis=ax[1, 0],
                              row_ids=np.arange(len(pc_fractions[121])), color=pc_fractions[121], title='121',
                              cmap=cat_cm, draw_cbar=False)
ax[1, 1] = plot_colored_cells(match_matrix=match_matrices[3]['115_1'], mouse_id=115, day='2022-08-12', axis=ax[1, 1],
                              row_ids=np.arange(len(pc_fractions[115])), color=pc_fractions[115], title='115',
                              cmap=cat_cm, draw_cbar=False)
ax[1, 2] = plot_colored_cells(match_matrix=match_matrices[4]['122_1'], mouse_id=122, day='2022-08-15', axis=ax[1, 2],
                              row_ids=np.arange(len(pc_fractions[122])), color=pc_fractions[122], title='122',
                              cmap=cat_cm, draw_cbar=False)





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


