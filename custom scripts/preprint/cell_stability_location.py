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
import datajoint as dj
from mpl_toolkits.axes_grid1 import make_axes_locatable

from schema import common_match, common_img, hheise_placecell, hheise_behav, common_mice, hheise_hist
from util import helper
from preprint import data_cleaning as dc

DAY_DIFF = 3
BACKWARDS = True

# Classes of possible PF locations
FINE_ZONES = np.array(['pre_RZ1', 'in_RZ1', 'RZ1-RZ2', 'in_RZ2', 'RZ2-RZ3', 'in_RZ3', 'RZ3-RZ4', 'in_RZ4', 'post_RZ4'])
RZ_MASK = np.array([False, True, False, True, False, True, False, True, False])

# Deficit
queries = ((common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=41'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=69' & 'day<="2021-03-23"'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=121' & 'day<"2022-09-09"'))

# Sham
queries = ((common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=115' & 'day<"2022-09-09"'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=122' & 'day<"2022-09-09"'))

# All
queries = ((common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=33'),   # 407 cells
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=41'),   # 140 cells
           # (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=63'),   # 87 cells
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=69' & 'day<="2021-03-23"'),  # 333 cells
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=85'),   # 229 cells
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=90'),   # 131 cells
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=93'),   # 397 cells
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=108' & 'day<"2022-09-09"'),  # 316 cells
           # (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=110' & 'day<"2022-09-09"'),  # 21 cells
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=114' & 'day<"2022-09-09"'),  # 307 cells
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=115' & 'day<"2022-09-09"'),  # 331 cells
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=121' & 'day<"2022-09-09"'),  # 791 cells
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=122' & 'day<"2022-09-09"')   # 401 cells
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

    # M63 has sessions with 170cm corridor, bin sizes dont match up. Skip analysis until fixed.
    # if np.unique(query.fetch('mouse_id'))[0] != 63:
    spatial_maps.append(query.get_matched_data(table=hheise_placecell.BinnedActivity.ROI, attribute='bin_spikerate',
                                               extra_restriction=dict(corridor_type=0, place_cell_id=2),
                                               return_array=True, relative_dates=True,
                                               surgery='Microsphere injection'))

    spat_dff_maps.append(query.get_matched_data(table=hheise_placecell.BinnedActivity.ROI, attribute='bin_activity',
                                                extra_restriction=dict(corridor_type=0, place_cell_id=2),
                                                return_array=True, relative_dates=True,
                                                surgery='Microsphere injection'))


#%% Save matched/fetched data to files for faster re-loading
import pickle
import os

dir = r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Data\analysis\matched_data'
with open(os.path.join(dir, 'match_matrices.pickle'), "wb") as output_file:
    pickle.dump(match_matrices, output_file)
with open(os.path.join(dir, 'is_pc.pickle'), "wb") as output_file:
    pickle.dump(is_pc, output_file)
with open(os.path.join(dir, 'pfs.pickle'), "wb") as output_file:
    pickle.dump(pfs, output_file)
with open(os.path.join(dir, 'spatial_maps.pickle'), "wb") as output_file:
    pickle.dump(spatial_maps, output_file)
with open(os.path.join(dir, 'spat_dff_maps.pickle'), "wb") as output_file:
    pickle.dump(spat_dff_maps, output_file)


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


def get_pf_location(match_matrix, spatial_dff, place_fields):
    """
    Compute center of mass of all place fields of matched cells of a single network.

    Args:
        match_matrix: single entry of match_matrices
        spatial_dff: single entry of spat_dff_maps
        place_fields: single entry of pfs

    Returns:

    """

    pf_com_df = np.array(match_matrix.copy(), dtype='object')
    pf_com_df[:] = np.nan
    # pf_com_sd_df = pf_com_df.copy()

    # Use Lambda function to check if a value is an empty list, or nan
    valid_sessions = np.where(~(pd.isna(place_fields) |
                                place_fields.applymap(lambda x: isinstance(x, list) and len(x) == 0)))

    for x, y in zip(valid_sessions[0], valid_sessions[1]):

        # Loop through multiple place fields
        pf_com_df[x, y] = [dc.place_field_com(spatial_dff[x, y], curr_pf)[0] for curr_pf in place_fields.iloc[x, y]]

        # pf_com_df[x, y] = [j[0] for j in curr_coms]
        # pf_com_sd_df[x, y] = [j[1] for j in curr_coms]

    pf_com_df = pd.DataFrame(data=pf_com_df, columns=match_matrix.columns, index=match_matrix.index)
    # pf_com_sd_df = pd.DataFrame(data=pf_com_sd_df, columns=match_matrix.columns, index=match_matrix.index)

    return pf_com_df


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
    pc_class[(pc_frac['pre_avg'] >= 0.3) & (pc_frac['early_post_avg'] >= 0.3)] = 3  # Class 3: PC pre and PC early post

    return pc_class


def plot_colored_cells(match_matrix, mouse_id, day, row_ids, color, cmap='magma', contour_thresh=0.05, draw_cbar=True,
                       background='avg_image', axis=None, title=None, cbar_ticklabels=None):

    # Fetch background image and initialize colormap
    username = 'hheise'
    key = dict(username=username, mouse_id=mouse_id, day=day)
    if background is None:
        bg = np.zeros((488, 488), dtype=int)   # Todo: dont hardcode FOV size
    else:
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


#%% Plot cells in the FOV, shaded by the correlation change in early poststroke compared to prestroke
fig, ax = plt.subplots(2, 3)

ax[0, 0] = plot_colored_cells(match_matrix=match_matrices[0]['41_1'], mouse_id=41, day='2020-08-27', axis=ax[0, 0],
                              row_ids=avg_dif[41].index, color=avg_dif[41]['early_post'], title='41', background=None)
ax[0, 1] = plot_colored_cells(match_matrix=match_matrices[1]['69_1'], mouse_id=69, day='2021-03-11', axis=ax[0, 1],
                              row_ids=avg_dif[69].index, color=avg_dif[69]['early_post'], title='69', background=None)
ax[0, 2].set_visible(False)
ax[1, 0] = plot_colored_cells(match_matrix=match_matrices[2]['121_1'], mouse_id=121, day='2022-08-15', axis=ax[1, 0],
                              row_ids=avg_dif[121].index, color=avg_dif[121]['early_post'], title='121', background=None)
ax[1, 1] = plot_colored_cells(match_matrix=match_matrices[3]['115_1'], mouse_id=115, day='2022-08-12', axis=ax[1, 1],
                              row_ids=avg_dif[115].index, color=avg_dif[115]['early_post'], title='115', background=None)
ax[1, 2] = plot_colored_cells(match_matrix=match_matrices[4]['122_1'], mouse_id=122, day='2022-08-15', axis=ax[1, 2],
                              row_ids=avg_dif[122].index, color=avg_dif[122]['early_post'], title='122', background=None)


## Correlate correlation_diff and avg_neighbour_diff
colormap = matplotlib.cm.get_cmap('viridis')
fig, ax = plt.subplots(2, 3)
sns.regplot(data=avg_neighbour_dif[41], x='value', y='neighbour_mean', ax=ax[0, 0], scatter_kws={'color': colormap(avg_neighbour_dif[41]['p'][~np.isnan(avg_neighbour_dif[41]['p'])])})
sns.regplot(data=avg_neighbour_dif[69], x='value', y='neighbour_mean', ax=ax[0, 1], scatter_kws={'color': colormap(avg_neighbour_dif[69]['p'][~np.isnan(avg_neighbour_dif[69]['p'])])})
sns.regplot(data=avg_neighbour_dif[121], x='value', y='neighbour_mean', ax=ax[1, 0], scatter_kws={'color': colormap(avg_neighbour_dif[121]['p'][~np.isnan(avg_neighbour_dif[121]['p'])])})
sns.regplot(data=avg_neighbour_dif[115], x='value', y='neighbour_mean', ax=ax[1, 1], scatter_kws={'color': colormap(avg_neighbour_dif[115]['p'][~np.isnan(avg_neighbour_dif[115]['p'])])})
sns.regplot(data=avg_neighbour_dif[122], x='value', y='neighbour_mean', ax=ax[1, 2], scatter_kws={'color': colormap(avg_neighbour_dif[122]['p'][~np.isnan(avg_neighbour_dif[122]['p'])])})

# Todo: Plot colorbars for p-value --> investigate why pvalue tracks nicely with y-axis

colormap = matplotlib.cm.get_cmap('viridis')
fig, ax = plt.subplots(2, 3)
sns.regplot(data=avg_neighbour_dif[41], x='value', y='neighbour_mean', ax=ax[0, 0]); ax[0, 0].set_title(41); ax[0, 0].set(xlabel='Stability', ylabel='Mean neighbourhood stability')
sns.regplot(data=avg_neighbour_dif[69], x='value', y='neighbour_mean', ax=ax[0, 1]); ax[0, 1].set_title(69); ax[0, 1].set(xlabel='Stability', ylabel='Mean neighbourhood stability')
ax[0, 2].set_visible(False)
sns.regplot(data=avg_neighbour_dif[121], x='value', y='neighbour_mean', ax=ax[1, 0]); ax[1, 0].set_title(121); ax[1, 0].set(xlabel='Stability', ylabel='Mean neighbourhood stability')
sns.regplot(data=avg_neighbour_dif[115], x='value', y='neighbour_mean', ax=ax[1, 1]); ax[1, 1].set_title(115); ax[1, 1].set(xlabel='Stability', ylabel='Mean neighbourhood stability')
sns.regplot(data=avg_neighbour_dif[122], x='value', y='neighbour_mean', ax=ax[1, 2]); ax[1, 2].set_title(122); ax[1, 2].set(xlabel='Stability', ylabel='Mean neighbourhood stability')


#%% Split cells into stable_pre-stable_post and unstable_pre-stable_post

pc_fractions = {41: compute_placecell_fractions(pc=is_pc[0]['41_1'][0], days=np.array(is_pc[0]['41_1'][1])),
                69: compute_placecell_fractions(pc=is_pc[1]['69_1'][0], days=np.array(is_pc[1]['69_1'][1])),
                121: compute_placecell_fractions(pc=is_pc[2]['121_1'][0], days=np.array(is_pc[2]['121_1'][1])),
                115: compute_placecell_fractions(pc=is_pc[3]['115_1'][0], days=np.array(is_pc[3]['115_1'][1])),
                122: compute_placecell_fractions(pc=is_pc[4]['122_1'][0], days=np.array(is_pc[4]['122_1'][1]))}


# Make categorical colormap
cat_cm = matplotlib.colors.ListedColormap(np.array([[0.4, 0.2, 0, 1],     # Class 0: brown
                                                    [0.0, 0.6, 1.0, 1],     # Class 1: blue
                                                    [0.2, 0.8, 0.2, 1],     # Class 2: green
                                                    [1.0, 0.4, 0.0, 1]]))   # Class 3: orange

fig, ax = plt.subplots(2, 3)
ax[0, 0] = plot_colored_cells(match_matrix=match_matrices[0]['41_1'], mouse_id=41, day='2020-08-27', axis=ax[0, 0],
                              row_ids=np.arange(len(pc_fractions[41])), color=pc_fractions[41], title='41', cmap=cat_cm,
                              draw_cbar=False, background='cor_image')
ax[0, 1] = plot_colored_cells(match_matrix=match_matrices[1]['69_1'], mouse_id=69, day='2021-03-11', axis=ax[0, 1],
                              row_ids=np.arange(len(pc_fractions[69])), color=pc_fractions[69], title='69', cmap=cat_cm,
                              cbar_ticklabels=['Never PC', 'Newly coding', 'Lost coding', 'Remaining PC'], background='cor_image')
ax[0, 2].set_visible(False)
ax[1, 0] = plot_colored_cells(match_matrix=match_matrices[2]['121_1'], mouse_id=121, day='2022-08-15', axis=ax[1, 0],
                              row_ids=np.arange(len(pc_fractions[121])), color=pc_fractions[121], title='121',
                              cmap=cat_cm, draw_cbar=False, background='cor_image')
ax[1, 1] = plot_colored_cells(match_matrix=match_matrices[3]['115_1'], mouse_id=115, day='2022-08-12', axis=ax[1, 1],
                              row_ids=np.arange(len(pc_fractions[115])), color=pc_fractions[115], title='115',
                              cmap=cat_cm, draw_cbar=False, background='cor_image')
ax[1, 2] = plot_colored_cells(match_matrix=match_matrices[4]['122_1'], mouse_id=122, day='2022-08-15', axis=ax[1, 2],
                              row_ids=np.arange(len(pc_fractions[122])), color=pc_fractions[122], title='122',
                              cmap=cat_cm, draw_cbar=False, background='cor_image')


#%% Compute phase of the periodic spatial map (position of maxima in each corridor quadrant) and plot in FOV


#%% Get fractions of PCs that prefer a certain location (inside/outside RZ, which RZ) --> unmatched data

def make_binary_column(df, col1, col2, lower, upper):
    return np.array(((df[col1].between(lower, upper)) | (df[col2].between(lower, upper))).astype(int))


zone_borders = (hheise_behav.CorridorPattern() & 'pattern="training"').rescale_borders(80)
# Move RZ a bit to the front, to compensate for visual mismatch/anticipation
zone_borders[:, 0] -= 1.33333
zone_borders[:, 1] -= 1.33333
# Only get data from normal, non-validation trials
pks = (hheise_placecell.PlaceCell() & 'corridor_type=0' & 'place_cell_id=2' & 'mouse_id!=63' & 'mouse_id!=38' & 'mouse_id!=89').fetch('KEY')
# Only get fully accepted place fields from accepted place cells
pf_key = dict(is_place_cell=1, large_enough=1, strong_enough=1, transients=1)

pc_location = []    # Preferred location of place cells. Can be above 100% due to PCs with more than 1 PF.
pf_location = []    # Location of place fields. Always adds up to 100%.
for pk in pks:

    # Get place fields of all accepted place cells of the session
    key, all_pfs = (hheise_placecell.PlaceCell.ROI * hheise_placecell.PlaceCell.PlaceField & pk & pf_key).fetch('KEY', 'bin_idx')

    # Compute day relative to surgery
    try:
        surg_day = (common_mice.Surgery & f'mouse_id={pk["mouse_id"]}' & 'surgery_type="Microsphere injection"').fetch1('surgery_date')
        rel_day = (pk['day'] - surg_day.date()).days
    except dj.errors.DataJointError:
        continue

    if len(key) <= 2:

        # Make summaries
        summary = {col: np.nan for col in FINE_ZONES}
        summary['in_RZ'] = -1
        summary['out_RZ'] = -1
        pc_location.append(pd.DataFrame(data=[dict(mouse_id=pk['mouse_id'], day=pk['day'], rel_day=rel_day, **summary)]))
        pf_location.append(pd.DataFrame(data=[dict(mouse_id=pk['mouse_id'], day=pk['day'], rel_day=rel_day, **summary)]))
        continue
    mask_pks, spat_maps = (hheise_placecell.BinnedActivity.ROI & key).get_normal_act('dff')

    if spat_maps.shape == (80,):
        continue

    pf = pd.DataFrame(key)

    # Compute Center of Mass for all place fields (some cells have more than 1 PF)
    pf_coms = []
    pf_sds = []
    for ind, k in enumerate(key):
        mask_idx = [i for i, dic in enumerate(mask_pks) if dic['mask_id'] == k['mask_id']][0]
        pf_com, pf_sd = dc.place_field_com(spat_maps[mask_idx], all_pfs[ind])
        pf_coms.append(pf_com)
        pf_sds.append(pf_sd)

    # pf_com = list(zip(*[dc.place_field_com(spat_map, pf_ind) for spat_map, pf_ind in zip(spat_maps, pfs)]))

    pf.drop(columns=['session_num', 'motion_id', 'caiman_id', 'place_cell_id', 'corridor_type'], inplace=True)
    pf['com'] = pf_coms
    pf['com_sd'] = pf_sds
    pf['com_min'] = pf['com'] - pf['com_sd']
    pf['com_max'] = pf['com'] + pf['com_sd']

    # Get position of each place field with respect to reward zones.
    prev_border = 0
    for idx, border in enumerate(zone_borders.flatten()):
        pf[FINE_ZONES[idx]] = pf['com'].between(prev_border, border)
        prev_border = border
    pf[FINE_ZONES[idx+1]] = pf['com'].between(prev_border, 80)

    # Add columns for in vs outside of RZs
    pf['in_RZ'] = pf[FINE_ZONES[RZ_MASK]].any(axis=1)
    pf['out_RZ'] = pf[FINE_ZONES[~RZ_MASK]].any(axis=1)

    # Make summaries
    n_cells = pf['mask_id'].nunique()
    summary = {col: pf[col].sum()/n_cells for col in FINE_ZONES}
    summary['in_RZ'] = pf['in_RZ'].sum()/n_cells
    summary['out_RZ'] = pf['out_RZ'].sum() / n_cells
    pc_location.append(pd.DataFrame(data=[dict(mouse_id=pk['mouse_id'], day=pk['day'], rel_day=rel_day, **summary)]))

    summary = {col: pf[col].sum()/len(pf) for col in FINE_ZONES}
    summary['in_RZ'] = pf['in_RZ'].sum()/len(pf)
    summary['out_RZ'] = pf['out_RZ'].sum() / len(pf)
    pf_location.append(pd.DataFrame(data=[dict(mouse_id=pk['mouse_id'], day=pk['day'], rel_day=rel_day, **summary)]))

pf_loc = pd.concat(pf_location)
pc_loc = pd.concat(pc_location)

# Export in/out data for line plots in Prism
pf_loc_export = pf_loc.pivot(index='rel_day', columns='mouse_id', values='in_RZ')
pf_loc_export.to_csv(r'C:\Users\hheise.UZH\Desktop\pc_location\pf_in_RZ.csv')
# Todo: finish analysing, plotting with Prism --> across mice, days


# Plot in_RZ fraction for all mice, color-coded by the number of spheres they have
spheres = pd.DataFrame((hheise_hist.MicrosphereSummary.Metric() &
                        'metric_name="spheres"').fetch('mouse_id', 'count_extrap', as_dict=True))
spheres.rename(columns={'count_extrap': 'spheres'}, inplace=True)
pf_loc_filt = pd.read_csv(r'C:\Users\hheise.UZH\Desktop\pc_location\pf_loc_smooth.csv')
pf_loc_filt = pd.melt(pf_loc_filt, id_vars='rel_day', var_name='mouse_id', value_name='in_RZ')
pf_loc_filt['mouse_id'] = pf_loc_filt['mouse_id'].astype(int)
pf_loc_sphere = pd.merge(pf_loc_filt, spheres, on='mouse_id')
sns.lineplot(data=pf_loc_sphere, x='rel_day', y='in_RZ', hue='spheres', hue_norm=matplotlib.colors.LogNorm())

# Correlate number of spheres and place cells in RZ (separately for different time periods)
total_mean = pf_loc_sphere.groupby('mouse_id')['in_RZ'].mean().rename('in_RZ_total')
pre_mean = pf_loc_sphere[pf_loc_sphere['rel_day'] <= 0].groupby('mouse_id')['in_RZ'].mean().rename('in_RZ_pre')
post_mean = pf_loc_sphere[pf_loc_sphere['rel_day'] > 0].groupby('mouse_id')['in_RZ'].mean().rename('in_RZ_post')
total_mean_sphere = pd.merge(spheres, total_mean, on='mouse_id').merge(pre_mean, on='mouse_id').merge(post_mean, on='mouse_id')

fig = plt.figure()
ax_total = sns.regplot(data=total_mean_sphere, x='spheres', y='in_RZ_total', logx=True, label='total')
ax_pre = sns.regplot(data=total_mean_sphere, x='spheres', y='in_RZ_pre', logx=True, label='pre')
ax_post = sns.regplot(data=total_mean_sphere, x='spheres', y='in_RZ_post', logx=True, label='post')
fig.legend()

total_mean_sphere.to_csv(r'C:\Users\hheise.UZH\Desktop\pc_location\in_RZ_spheres.csv')

#%% Periodicity

"""
1. Get classification of which cells are PCs across whole experiment
2. Get preferred location of these PCs for each session (global, quadrant, binary)
3. Check influence
    a. Does a larger fraction of PCs in poststroke than prestroke have a similar preferred location like the stable cells?
    b. Do the newly coding PCs share more often than expected (how to test?) a preferred location with the stable cells?
"""

def measure_cell_influence(cell_class, pf_center, mouse_id, match_matrix, days, exclude_other_class3_cells=False,
                           rz_border_buffer=0):

    # Transform PF centers into quadrant coordinates (distance to next RZ start) -> applied to each pf center
    def get_distance(centers, borders):

        def comp_dist(cent, b):
            dist = b - cent
            return np.min(dist[dist > 0])  # Get distance to next RZ (only positive distances)

        try:
            return [comp_dist(c, borders) for c in centers]
        except TypeError:
            return comp_dist(centers, borders)

    # Transform PF centers into binary values (in RZ or not) -> applied to each pf center
    def in_reward_zone(centers, borders):
        try:
            return [any([b[0] <= c <= b[1] for b in borders]) for c in centers]
        except TypeError:
            return any([b[0] <= centers <= b[1] for b in borders])

    def get_zone(centers, borders):

        def single_zone(cent, b):
            # Get fine-grained location of PF
            cent_loc = []
            prev_bord = 0
            for i, bor in enumerate(b.flatten()):
                cent_loc.append(prev_bord <= cent < bor)
                prev_bord = bor
            cent_loc.append(prev_bord <= cent <= 80)
            c_loc = np.where(cent_loc)[0]
            if len(c_loc) != 1:
                return 'ERROR'
            else:
                return fine_zones[c_loc[0]]

        fine_zones = np.array(['pre_RZ1', 'in_RZ1', 'RZ1-RZ2', 'in_RZ2', 'RZ2-RZ3', 'in_RZ3',
                               'RZ3-RZ4', 'in_RZ4', 'post_RZ4'])

        try:
            return [single_zone(c, borders) for c in centers]
        except TypeError:
            return single_zone(centers, borders)

    # print(mouse_id)

    # Only take accepted PFs from accepted PCs
    pf_k = dict(place_cell_id=2, corridor_type=0, is_place_cell=1, large_enough=1, strong_enough=1, transients=1)
    bord = (hheise_behav.CorridorPattern() & 'pattern="training"').rescale_borders(80)
    bord[:, 0] -= rz_border_buffer
    bord[:, 1] += rz_border_buffer

    # print('Using borders', bord)

    # For all class-3 cells, compute fraction of other place cells that are similar/average distance of place fields
    # all sessions where class-3 cells are actually place cells
    class3 = np.where(cell_class == 3)[0]
    class3_df = []
    for day, relative_day in zip(pf_center.columns, days):

        # If there are class-3-cells in this session, start loading data for all place cells
        if not pf_center[day].iloc[class3].isna().all():

            # Get dataframe indices of all class3-cells that are place cells in the current session
            curr_active_class3 = pf_center[day].iloc[class3].index[np.where(~pf_center[day].iloc[class3].isna())[0]]
            # Use these to get the global mask_id for the class3 neurons in the current session
            real_curr_class3_ids = match_matrix[day][curr_active_class3].values

            all_curr_pfs = pd.DataFrame((hheise_placecell.PlaceCell.ROI * hheise_placecell.PlaceCell.PlaceField &
                                         f'mouse_id={mouse_id}' & pf_k &
                                         common_match.MatchedIndex().string2key(title=day)).fetch('KEY', 'bin_idx', as_dict=True))

            # Compute PF CoM for all place cells of the current session
            dff_pks, all_spat_dff = (hheise_placecell.BinnedActivity.ROI & all_curr_pfs).get_normal_act(trace='dff', return_pks=True)
            all_spat = pd.DataFrame(dff_pks)
            if len(dff_pks) == 1:
                all_spat['spat_dff'] = [all_spat_dff]
            else:
                all_spat['spat_dff'] = list(all_spat_dff)
            all_pc = pd.merge(all_curr_pfs, all_spat)
            all_pf_com = [dc.place_field_com(dff, bin_idx)[0] for bin_idx, dff in zip(all_pc['bin_idx'], all_pc['spat_dff'])]
            all_pc['com'] = all_pf_com
            # Make a dummy 5th border (same distance as RZ1-RZ2) for place fields after the 4th RZ
            all_pc['pf_quad'] = all_pc['com'].apply(get_distance, borders=np.append(bord[:, 0], bord[-1, 0]+(bord[1, 0]-bord[0, 0])))
            all_pc['pf_rz'] = all_pc['com'].apply(in_reward_zone, borders=bord)
            all_pc['pf_zone'] = all_pc['com'].apply(get_zone, borders=bord)

            if exclude_other_class3_cells:
                all_class3_cells = all_pc[all_pc['mask_id'].isin(real_curr_class3_ids)]
                all_pc = all_pc[~all_pc['mask_id'].isin(real_curr_class3_ids)]

            for df_idx, dj_mask_id in zip(curr_active_class3, real_curr_class3_ids):
                # for each currently PC class3 cell, compute distance/similarity of the c3 cell to all other PCs
                # (also non-tracked) at that session
                # print(df_idx, dj_mask_id)

                if exclude_other_class3_cells:
                    # If we excluded all class3 cells before, we dont have to do it for each cell
                    curr_class3 = all_class3_cells[all_class3_cells['mask_id'] == dj_mask_id]
                    rest_pc = all_pc
                else:
                    # Otherwise, remove the current class 3 cell from the DF for processing
                    curr_class3 = all_pc[all_pc['mask_id'] == dj_mask_id]
                    rest_pc = all_pc[all_pc['mask_id'] != dj_mask_id]

                # curr_class3 contains all PFs of the current stable PC. Make separate analysis for all accepted PFs.
                # Treating multiple PFs of the same PC like separate PCs makes sense, since the hypothesis is that the
                # stable cells drive other cells to be the same, which would also mean having the same number of PFs.
                # However, if a stable cell has 2 PFs, and another cell has the same 2 PFs, this analysis would give a
                # bad score, since the two PFs might be different from each other. Todo: Figure out how to deal with this.
                for i, pf_row in curr_class3.iterrows():

                    class3_df.append(pd.DataFrame(
                        [dict(
                            # General info about the current class3 place cell/field, to find it again in the database
                            df_label_idx=df_idx, df_integer_idx=np.where(match_matrix[day].index == df_idx)[0][0],
                            day=day, rel_day=relative_day, mask_idx=int(dj_mask_id), pf_id=pf_row['place_field_id'],
                            # Attributes of the class3 field that were compared to the rest of the place cells
                            n_rest_pc=len(rest_pc), com=pf_row['com'],
                            pf_quad=pf_row['pf_quad'], pf_rz=pf_row['pf_rz'], pf_zone=pf_row['pf_zone'],
                            # Comparison of class3 field vs other PCs
                            avg_dist=(rest_pc['com'] - pf_row['com']).abs().mean() * (400/80),               # Average distance of PF CoM to other accepted PFs in cm
                            avg_quad_dist=(rest_pc['pf_quad'] - pf_row['pf_quad']).abs().mean() * (400/80),  # Average distance of PF CoM to other accepted PFs, normalized by quadrants, in cm
                            share_rz=(rest_pc['pf_rz'] == pf_row['pf_rz']).sum()/len(rest_pc),      # Fraction of PFs that are also in a reward zone/not in reward zone
                            share_zone=(rest_pc['pf_zone'] == pf_row['pf_zone']).sum()/len(rest_pc),
                        )
                        ]))
    class3_df = pd.concat(class3_df, ignore_index=True)
    # class3_df.sort_values(by=['df_label_idx', 'rel_day', 'pf_id'], inplace=True)    # sort final DF by Dataframe index/global matched_matrix ID (easier to follow up single cells)
    class3_df.sort_values(by=['rel_day', 'df_label_idx', 'pf_id'], inplace=True)    # sort final DF by date (easier to plot time course)
    return class3_df


def measure_newly_coding_influence(cell_class, pf_center, mouse_id, match_matrix, days):

    # Transform PF centers into quadrant coordinates (distance to next RZ start) -> applied to each pf center
    def get_distance(centers, borders):
        # Distance to next RZ start: Minimal right before the start, maximum right after the start
        # The resulting quadrant goes from max_dist (in RZ) -> min_dist (before next RZ)
        def comp_dist(cent, b):
            dist = b - cent
            return np.min(dist[dist > 0])  # Get distance to next RZ (only positive distances)

        try:
            return [comp_dist(c, borders) for c in centers]
        except TypeError:
            return comp_dist(centers, borders)

    # Transform PF centers into binary values (in RZ or not) -> applied to each pf center
    def in_reward_zone(centers, borders):
        try:
            return [any([b[0] <= c <= b[1] for b in borders]) for c in centers]
        except TypeError:
            return any([b[0] <= centers <= b[1] for b in borders])

    def get_zone(centers, borders):

        def single_zone(cent, b):
            # Get fine-grained location of PF
            cent_loc = []
            prev_bord = 0
            for i, bor in enumerate(b.flatten()):
                cent_loc.append(prev_bord <= cent < bor)
                prev_bord = bor
            cent_loc.append(prev_bord <= cent <= 80)
            c_loc = np.where(cent_loc)[0]
            if len(c_loc) != 1:
                return 'ERROR'
            else:
                return FINE_ZONES[c_loc[0]]

        try:
            return [single_zone(c, borders) for c in centers]
        except TypeError:
            return single_zone(centers, borders)

    # print(mouse_id)

    # Only take accepted PFs from accepted PCs
    pf_k = dict(place_cell_id=2, corridor_type=0, is_place_cell=1, large_enough=1, strong_enough=1, transients=1)
    bord = (hheise_behav.CorridorPattern() & 'pattern="training"').rescale_borders(80)

    # For all class-3 cells, compute fraction of other place cells that are similar/average distance of place fields
    # all sessions where class-3 cells are actually place cells
    class3 = np.where(cell_class == 3)[0]
    class1 = np.where(cell_class == 1)[0]       # Newly coding cells (never PC pre, at least once PC early post)
    class3_df = []
    for day, relative_day in zip(pf_center.columns, days):

        # Only process poststroke sessions
        if relative_day < 1:
            continue

        # If there are class-3-cells in this session, start loading data for all place cells
        if not pf_center[day].iloc[class3].isna().all():

            # Get dataframe indices of all class3-cells that are place cells in the current session
            curr_active_class3 = pf_center[day].iloc[class3].index[np.where(~pf_center[day].iloc[class3].isna())[0]]
            # Use these to get the global mask_id for the class3 neurons in the current session
            real_curr_class3_ids = match_matrix[day][curr_active_class3].values

            # Get dataframe label indices of all class-1 cells that are place cells in the current session
            curr_active_class1 = pf_center[day].iloc[class1].index[np.where(~pf_center[day].iloc[class1].isna())[0]]
            # Use these to get the global mask_id for the class3 neurons in the current session
            real_curr_class1_ids = match_matrix[day][curr_active_class1].values

            # Compute PF CoM for all (tracked) place cells of the current session
            all_pc = pd.DataFrame(dict(day=day, rel_day=relative_day, mask_id=match_matrix[day][~pf_center[day].isna()],
                                       com=pf_center[day][~pf_center[day].isna()]))
            # Query database for correct place_field_id
            pf_ids = pd.DataFrame((hheise_placecell.PlaceCell.PlaceField & f'mouse_id={mouse_id}' & pf_k &
                                   common_match.MatchedIndex().string2key(title=day) &
                                   f'mask_id in {helper.in_query(all_pc["mask_id"])}'
                                   ).fetch('mask_id', 'place_field_id', as_dict=True))
            # Match place fields to place field IDs (merge does not work for PCs with more than 1 PF)
            all_pc['place_field_id'] = [pf_ids[pf_ids['mask_id'] == row['mask_id']]['place_field_id'].values
                                        for i, row in all_pc.iterrows()]
            all_pc = all_pc.explode(['com', 'place_field_id'])
            # Make a dummy 5th border (same distance as RZ1-RZ2) for place fields after the 4th RZ
            all_pc['pf_quad'] = all_pc['com'].apply(get_distance, borders=np.append(bord[:, 0], bord[-1, 0]+(bord[1, 0]-bord[0, 0])))
            all_pc['pf_rz'] = all_pc['com'].apply(in_reward_zone, borders=bord)
            all_pc['pf_zone'] = all_pc['com'].apply(get_zone, borders=bord)

            # Extract data for class3 and class1 cells
            class3_data = all_pc.loc[curr_active_class3]
            class1_data = all_pc.loc[curr_active_class1]

            # Skip session if there are no newly coding place cells
            if len(class1_data) == 0:
                continue

            for int_idx, (row_idx, row) in enumerate(class3_data.iterrows()):

                class3_df.append(pd.DataFrame(
                    [dict(
                        # General info about the current class3 place cell/field, to find it again in the database
                        df_label_idx=row_idx, df_integer_idx=int_idx, day=day, rel_day=relative_day,
                        mask_idx=row['mask_id'], pf_id=row['place_field_id'],
                        # Attributes of the class3 field that were compared to the rest of the place cells
                        n_rest_pc=len(class1_data), com=row['com'],
                        pf_quad=row['pf_quad'], pf_rz=row['pf_rz'], pf_zone=row['pf_zone'],
                        # Comparison of class3 field vs other PCs
                        avg_dist=(class1_data['com'] - row['com']).abs().mean() * (400/80),               # Average distance of PF CoM to newly coding PFs in cm
                        avg_quad_dist=(class1_data['pf_quad'] - row['pf_quad']).abs().mean() * (400/80),  # Average distance of PF CoM to newly coding PFs, normalized by quadrants, in cm
                        share_rz=(class1_data['pf_rz'] == row['pf_rz']).sum()/len(class1_data),           # Fraction of newly coding PFs that are in the same area as the class3 cell
                        share_zone=(class1_data['pf_zone'] == row['pf_zone']).sum()/len(class1_data),
                    )
                    ]))
    class3_df = pd.concat(class3_df, ignore_index=True)
    # class3_df.sort_values(by=['df_label_idx', 'rel_day', 'pf_id'], inplace=True)    # sort final DF by Dataframe index/global matched_matrix ID (easier to follow up single cells)
    class3_df.sort_values(by=['rel_day', 'df_label_idx', 'pf_id'], inplace=True)    # sort final DF by date (easier to plot time course)
    return class3_df

# 1.
pc_fractions = {41: compute_placecell_fractions(pc=is_pc[0]['41_1'][0], days=np.array(is_pc[0]['41_1'][1])),
                69: compute_placecell_fractions(pc=is_pc[1]['69_1'][0], days=np.array(is_pc[1]['69_1'][1])),
                121: compute_placecell_fractions(pc=is_pc[2]['121_1'][0], days=np.array(is_pc[2]['121_1'][1])),
                115: compute_placecell_fractions(pc=is_pc[3]['115_1'][0], days=np.array(is_pc[3]['115_1'][1])),
                122: compute_placecell_fractions(pc=is_pc[4]['122_1'][0], days=np.array(is_pc[4]['122_1'][1]))}

pc_fractions = {33: compute_placecell_fractions(pc=is_pc[0]['33_1'][0], days=np.array(is_pc[0]['33_1'][1])),
                63: compute_placecell_fractions(pc=is_pc[4]['63_1'][0], days=np.array(is_pc[4]['63_1'][1])),
                85: compute_placecell_fractions(pc=is_pc[1]['85_1'][0], days=np.array(is_pc[1]['85_1'][1])),
                90: compute_placecell_fractions(pc=is_pc[2]['90_1'][0], days=np.array(is_pc[2]['90_1'][1])),
                93: compute_placecell_fractions(pc=is_pc[3]['93_1'][0], days=np.array(is_pc[3]['93_1'][1])),}
                # 108: compute_placecell_fractions(pc=is_pc[4]['108_1'][0], days=np.array(is_pc[4]['108_1'][1])),
                # 114: compute_placecell_fractions(pc=is_pc[5]['114_1'][0], days=np.array(is_pc[5]['114_1'][1]))}

pc_fractions = {33: compute_placecell_fractions(pc=is_pc[0]['33_1'][0], days=np.array(is_pc[0]['33_1'][1])),
                41: compute_placecell_fractions(pc=is_pc[1]['41_1'][0], days=np.array(is_pc[1]['41_1'][1])),
                # 63: compute_placecell_fractions(pc=is_pc[4]['63_1'][0], days=np.array(is_pc[4]['63_1'][1])),
                69: compute_placecell_fractions(pc=is_pc[2]['69_1'][0], days=np.array(is_pc[2]['69_1'][1])),
                85: compute_placecell_fractions(pc=is_pc[3]['85_1'][0], days=np.array(is_pc[3]['85_1'][1])),
                90: compute_placecell_fractions(pc=is_pc[4]['90_1'][0], days=np.array(is_pc[4]['90_1'][1])),
                93: compute_placecell_fractions(pc=is_pc[5]['93_1'][0], days=np.array(is_pc[5]['93_1'][1])),
                121: compute_placecell_fractions(pc=is_pc[7]['121_1'][0], days=np.array(is_pc[7]['121_1'][1])),
                115: compute_placecell_fractions(pc=is_pc[6]['115_1'][0], days=np.array(is_pc[6]['115_1'][1])),
                122: compute_placecell_fractions(pc=is_pc[8]['122_1'][0], days=np.array(is_pc[8]['122_1'][1]))}
                # 108: compute_placecell_fractions(pc=is_pc[4]['108_1'][0], days=np.array(is_pc[4]['108_1'][1])),
                # 114: compute_placecell_fractions(pc=is_pc[5]['114_1'][0], days=np.array(is_pc[5]['114_1'][1]))}

# 2.
pf_centers = {41: get_pf_location(match_matrix=match_matrices[0]['41_1'], spatial_dff=spat_dff_maps[0]['41_1'][0], place_fields=pfs[0]['41_1'][0]),
              69: get_pf_location(match_matrix=match_matrices[1]['69_1'], spatial_dff=spat_dff_maps[1]['69_1'][0], place_fields=pfs[1]['69_1'][0]),
              121: get_pf_location(match_matrix=match_matrices[2]['121_1'], spatial_dff=spat_dff_maps[2]['121_1'][0], place_fields=pfs[2]['121_1'][0]),
              115: get_pf_location(match_matrix=match_matrices[3]['115_1'], spatial_dff=spat_dff_maps[3]['115_1'][0], place_fields=pfs[3]['115_1'][0]),
              122: get_pf_location(match_matrix=match_matrices[4]['122_1'], spatial_dff=spat_dff_maps[4]['122_1'][0], place_fields=pfs[4]['122_1'][0])}

pf_centers = {33: get_pf_location(match_matrix=match_matrices[0]['33_1'], spatial_dff=spat_dff_maps[0]['33_1'][0], place_fields=pfs[0]['33_1'][0]),
              41: get_pf_location(match_matrix=match_matrices[1]['41_1'], spatial_dff=spat_dff_maps[1]['41_1'][0], place_fields=pfs[1]['41_1'][0]),
              69: get_pf_location(match_matrix=match_matrices[2]['69_1'], spatial_dff=spat_dff_maps[2]['69_1'][0], place_fields=pfs[2]['69_1'][0]),
              # 63: get_pf_location(match_matrix=match_matrices[1]['63_1'], spatial_dff=spat_dff_maps[1]['63_1'][0], place_fields=pfs[1]['69_1'][0]),
              85: get_pf_location(match_matrix=match_matrices[3]['85_1'], spatial_dff=spat_dff_maps[3]['85_1'][0], place_fields=pfs[3]['85_1'][0]),
              90: get_pf_location(match_matrix=match_matrices[4]['90_1'], spatial_dff=spat_dff_maps[4]['90_1'][0], place_fields=pfs[4]['90_1'][0]),
              93: get_pf_location(match_matrix=match_matrices[5]['93_1'], spatial_dff=spat_dff_maps[5]['93_1'][0], place_fields=pfs[5]['93_1'][0]),
              121: get_pf_location(match_matrix=match_matrices[7]['121_1'], spatial_dff=spat_dff_maps[7]['121_1'][0], place_fields=pfs[7]['121_1'][0]),
              115: get_pf_location(match_matrix=match_matrices[6]['115_1'], spatial_dff=spat_dff_maps[6]['115_1'][0], place_fields=pfs[6]['115_1'][0]),
              122: get_pf_location(match_matrix=match_matrices[8]['122_1'], spatial_dff=spat_dff_maps[8]['122_1'][0], place_fields=pfs[8]['122_1'][0])}
              # 108: get_pf_location(match_matrix=match_matrices[4]['108_1'], spatial_dff=spat_dff_maps[4]['108_1'][0], place_fields=pfs[4]['108_1'][0]),
              # 114: get_pf_location(match_matrix=match_matrices[5]['114_1'], spatial_dff=spat_dff_maps[5]['114_1'][0], place_fields=pfs[5]['114_1'][0])}

# 3.a
influence_dict = {k: measure_cell_influence(cell_class=pc_fractions[k], pf_center=pf_centers[k], mouse_id=k,
                                            match_matrix=match_matrices[i][f'{k}_1'], days=pfs[i][f'{k}_1'][1],
                                            rz_border_buffer=0)
                  for i, k in enumerate(pf_centers)}
influence = pd.concat([df.assign(mouse_id=key) for key, df in influence_dict.items()], ignore_index=True)
influence_long = influence.melt(id_vars=influence.columns[~influence.columns.isin(['avg_dist', 'avg_quad_dist',
                                                                                   'share_rz', 'share_zone'])])

# 3.b
newly_coding_dict = {k: measure_newly_coding_influence(cell_class=pc_fractions[k], pf_center=pf_centers[k], mouse_id=k,
                                                       match_matrix=match_matrices[i][f'{k}_1'], days=pfs[i][f'{k}_1'][1])
                     for i, k in enumerate(pf_centers)}
new_coding = pd.concat([df.assign(mouse_id=key) for key, df in newly_coding_dict.items()], ignore_index=True)
new_coding_long = new_coding.melt(id_vars=new_coding.columns[~new_coding.columns.isin(['avg_dist', 'avg_quad_dist',
                                                                                      'share_rz', 'share_zone'])])

# Plotting Influence and Newly Coding overview FacetGrids
g = sns.FacetGrid(influence_long[influence_long['variable'] != 'share_zone'], col='mouse_id', row='variable',
                  sharey='row', col_order=influence_dict.keys(), margin_titles=True)
for (row_val, col_val), ax in g.axes_dict.items():
    ax.axvline(0.5, linestyle='--', c='r')
g.map_dataframe(sns.lineplot, x='rel_day', y='value')
g.set_titles(col_template="M{col_name}", row_template="{row_name}")


# Get number of class 3 cells that are focussed on reward zones for each mouse
# How often are class 3 cells focussed on reward zones, per mouse
class3_rz = []
for mouse in influence['mouse_id'].unique():
    curr_mouse = influence[influence['mouse_id'] == mouse]

    for cell_id in curr_mouse['df_label_idx'].unique():
        curr_cell = curr_mouse[curr_mouse['df_label_idx'] == cell_id]

        occurrence_pre = (curr_cell['rel_day'] <= 0).sum()
        rz_pre = curr_cell[curr_cell['rel_day'] <= 0]['pf_rz'].sum()
        pre_frac = rz_pre/occurrence_pre

        occurrence_early = ((curr_cell['rel_day'] > 0) & (curr_cell['rel_day'] <= 9)).sum()
        rz_early = curr_cell[(curr_cell['rel_day'] > 0) & (curr_cell['rel_day'] <= 9)]['pf_rz'].sum()
        early_frac = rz_early/occurrence_early

        occurrence_late = (curr_cell['rel_day'] > 9).sum()
        rz_late = curr_cell[curr_cell['rel_day'] > 9]['pf_rz'].sum()
        late_frac = rz_late/occurrence_late

        class3_rz.append(pd.DataFrame([dict(mouse_id=mouse, df_label_idx=cell_id,
                                            occ_pre=occurrence_pre, rz_pre=rz_pre, pre_frac=pre_frac*100,
                                            occ_early=occurrence_early, rz_early=rz_early, early_frac=early_frac*100,
                                            occ_late=occurrence_late, rz_late=rz_late, late_frac=late_frac*100)]))
class3_rz = pd.concat(class3_rz)

class3_rz['occ_post'] = class3_rz['occ_early'] + class3_rz['occ_late']
class3_rz['rz_post'] = class3_rz['rz_early'] + class3_rz['rz_late']
class3_rz['post_frac'] = class3_rz['rz_post'] / class3_rz['occ_post']
class3_rz['occ_total'] = class3_rz['occ_pre'] + class3_rz['occ_post']
class3_rz['rz_total'] = class3_rz['rz_pre'] + class3_rz['rz_post']
class3_rz['total_frac'] = class3_rz['rz_total'] / class3_rz['occ_total']

class3_rz.to_csv(r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Papers\preprint\figure3\pc_location\class3_rz_fractions_buffer.csv')

# Transform to Prism format (nested columns)
arrs = []
mice = []
for mouse in class3_rz['mouse_id'].unique():
    mice.append(mouse)
    curr_mouse = class3_rz[class3_rz['mouse_id'] == mouse][['pre_frac', 'early_frac', 'late_frac']]
    curr_mouse = curr_mouse.rename(columns={col: str(mouse) + '_' + str(col) for col in curr_mouse.columns})
    arrs.append(curr_mouse)

arrs = [df.reset_index(drop=True) for df in arrs]
nested = pd.concat(arrs, axis=1)
nested.to_csv(r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Papers\preprint\figure3\pc_location\class3_rz_fractions_buffer_prism.csv')


#%% Characterize class 3 place cells

def characterize_class3(influence_df, rz_border_buffer, mouse_order=None):

    def draw_heatmap(dataset, ax_obj, cbar_pad=0.02):
        ax_obj = sns.heatmap(dataset, ax=ax_obj, cmap='magma', cbar_kws={'label': '% per session', 'pad': cbar_pad})
        ax_obj.invert_yaxis()
        return ax_obj

    def add_histplot(dataset, ax_obj, yvar, nbins, max_bin, kde_smoothing=1.0):
        divider = make_axes_locatable(ax_obj)  # Create a divider for the existing axes
        new_ax = divider.append_axes('right', size='20%', pad=0.1)  # Add a new axes to the right of the main axes
        new_ax = sns.histplot(dataset, y=yvar, ax=new_ax, bins=nbins, binrange=(0, max_bin),
                              stat='density', color='lightblue')
        new_ax = sns.kdeplot(dataset, y=yvar, ax=new_ax, bw_adjust=kde_smoothing)
        new_ax.spines[['right', 'top', 'bottom']].set_visible(False)
        new_ax.set(xticks=[], xlabel='', yticks=[], ylabel='', ylim=(0, max_bin))
        return new_ax

    def plot_coms(data_df, axis, m_id, var, num_cells):

        n_bins = 120
        zones = (hheise_behav.CorridorPattern() & 'pattern="training"').rescale_borders(n_bins)

        # Remap relative days to a contiguous integer scale to remove gaps between poststroke days
        remapped_values = {value: index for index, value in enumerate(data_df['rel_day'].unique())}
        remapped_days = np.array([remapped_values[value] for value in data_df['rel_day']])
        data_df['x'] = remapped_days

        # Make custom heatmap to get percentages per x-bin (day)
        com_heat = np.zeros((n_bins, len(data_df['x'].unique())))
        for d in data_df['x'].unique():
            histo = np.histogram(data_df[data_df['x'] == d][var], bins=n_bins, range=(0, 80), density=True)[0]
            com_heat[:, d] = (histo/np.sum(histo))*100

        # plt.figure()
        # axis = plt.subplot(111)
        axis = draw_heatmap(com_heat, axis)

        # Draw a KDE-Histogram on the side
        add_histplot(data_df, axis, var, n_bins, 80, 0.2)

        # Draw reward zone borders
        for z in zones:
            axis.axhline(z[0], linestyle='--', c='green')
            axis.axhline(z[1], linestyle='--', c='green')

        # Set X-ticks to relative days and draw red line at smallest positive day for Microsphere injection
        axis.set_xticklabels(axis.get_xticks(), rotation=0)
        axis.set(xticks=data_df['x'].unique()+0.5, xticklabels=data_df['rel_day'].unique())
        axis.axvline(np.where(data_df['rel_day'].unique() > 0)[0][0], linestyle='--', c='red')
        axis.set_title(f'M{m_id} - {num_cells} "Always-PC" cells')
        # # Using seaborns default histplot (normalizes across whole heatmap)
        # plt.figure()
        # axis = plt.subplot(111)
        # out = sns.histplot(coms, x='dummy_x', y='com', ax=axis, bins=120, discrete=(True, False), thresh=None,
        #                    stat='percent', binrange=(None, (0, 80)), cmap='magma', cbar=True)
        # out = axis.set(xticks=np.arange(len(coms['rel_day'].unique())), xticklabels=coms['rel_day'].unique())

        return axis

    def plot_quadrants(data_df, axis, m_id, var, num_cells):

        # Compute total quadrant length (distance from one RZ start to the next)
        size_factor = 120/80        # Size factor between VR units (integer bins of RZ zones) and spatial map units (5 cm)
        zones = (hheise_behav.CorridorPattern() & 'pattern="training"').rescale_borders(80)
        raw_quadrant_length = np.mean(zones[1:, 0] - zones[:-1, 0])
        quadrant_length = int(np.round(raw_quadrant_length * size_factor))
        zone_length = int(np.round(np.mean(zones[:, 1] - zones[:, 0]) * size_factor))

        # Remap relative days to a contiguous integer scale to remove gaps between poststroke days
        remapped_values = {value: index for index, value in enumerate(data_df['rel_day'].unique())}
        remapped_days = np.array([remapped_values[value] for value in data_df['rel_day']])
        data_df['x'] = remapped_days

        # Make custom heatmap to get percentages per x-bin (day)
        com_heat = np.zeros((quadrant_length, len(data_df['x'].unique())))
        for d in data_df['x'].unique():
            histo = np.histogram(data_df[data_df['x'] == d][var], bins=quadrant_length, range=(0, raw_quadrant_length),
                                 density=True)[0]
            com_heat[:, d] = (histo/np.sum(histo))*100

        # plt.figure()
        # axis = plt.subplot(111)
        axis = sns.heatmap(com_heat, ax=axis, cmap='magma', cbar_kws={'label': '% per session', 'pad': 0.02})
        # axis.invert_yaxis()       # Quadrant location is distance to next RZ -> inverse position (gets smaller along the corridor), so no axis reversion is needed)


        # Draw a KDE-Histogram on the side
        hist_ax = add_histplot(data_df, axis, var, quadrant_length, raw_quadrant_length, 0.75)
        hist_ax.invert_yaxis()  # Y-axis has to be inverted, since sns.heatmap() inverts it as well

        # Draw reward zone borders
        axis.axhline(quadrant_length-zone_length, linestyle='--', c='green', linewidth=3)

        # Set X-ticks to relative days and draw red line at smallest positive day for Microsphere injection
        axis.set_xticklabels(axis.get_xticks(), rotation=0)
        axis.set_yticklabels(axis.get_yticks(), rotation=0)
        axis.set(yticks=np.arange(quadrant_length)[::2]+0.5, yticklabels=np.arange(quadrant_length)[::2])
        axis.set(xticks=data_df['x'].unique() + 0.5, xticklabels=data_df['rel_day'].unique())
        axis.axvline(np.where(data_df['rel_day'].unique() > 0)[0][0], linestyle='--', c='red')
        axis.set_title(f'M{m_id} - {num_cells} "Always-PC" cells')
        axis.set_ylabel('Distance to next RZ')

        return axis

    def plot_zones(data_df, axis, m_id, var, num_cells):

        # Remap relative days to a contiguous integer scale to remove gaps between poststroke days
        remapped_values = {value: index for index, value in enumerate(data_df['rel_day'].unique())}
        remapped_days = np.array([remapped_values[value] for value in data_df['rel_day']])
        data_df['x'] = remapped_days

        # Make custom heatmap to get percentages per x-bin (day)
        com_heat = np.zeros((len(FINE_ZONES), len(data_df['x'].unique())))
        for d in data_df['x'].unique():
            # Count occurrence of each zone in the current session
            zone_counts = np.array([np.sum((data_df[data_df['x'] == d][var] == z)) for z in FINE_ZONES])
            com_heat[:, d] = (zone_counts / np.sum(zone_counts)) * 100

        # plt.figure()
        # axis = plt.subplot(111)
        axis = sns.heatmap(com_heat, ax=axis, cmap='magma', cbar_kws={'label': '% per session', 'pad': 0.02})
        axis.invert_yaxis()

        # Draw a barplot of the zone counts on the side
        divider = make_axes_locatable(axis)  # Create a divider for the existing axes
        new_ax = divider.append_axes('right', size='20%', pad=0.1)  # Add a new axes to the right of the main axes
        # Use a reversed dark green color palette
        sns.barplot(y=FINE_ZONES, x=np.array([np.sum((data_df[var] == z)) for z in FINE_ZONES]), ax=new_ax,
                    width=1, palette=reversed(sns.color_palette("Greens_d", len(FINE_ZONES))))
        new_ax.invert_yaxis()
        new_ax.spines[['right', 'top', 'bottom']].set_visible(False)
        new_ax.set(xticks=[], xlabel='', yticks=[], ylabel='')

        # Set X-ticks to relative days and draw red line at smallest positive day for Microsphere injection
        axis.set_xticklabels(axis.get_xticks(), rotation=0)
        axis.set_yticklabels(axis.get_yticks(), rotation=0)
        axis.set(yticks=np.arange(len(FINE_ZONES)) + 0.5, yticklabels=FINE_ZONES)
        axis.set(xticks=data_df['x'].unique() + 0.5, xticklabels=data_df['rel_day'].unique())
        axis.axvline(np.where(data_df['rel_day'].unique() > 0)[0][0], linestyle='--', c='red')
        axis.set_title(f'M{m_id} - {num_cells} "Always-PC" cells')

        return axis

    def plot_rzs(data_df, axis, m_id, var, num_cells):

        # Get fraction of corridor occupied by RZs
        zones = (hheise_behav.CorridorPattern() & 'pattern="training"').rescale_borders(120)
        # *1.5 because here corridor is binned to 120 instead of 80 bins, making each bin 50% smaller
        zones[:, 0] -= rz_border_buffer * 1.5
        zones[:, 1] += rz_border_buffer * 1.5
        total_zones = np.sum(zones[:, 1] - zones[:, 0]) / 120

        # Remap relative days to a contiguous integer scale to remove gaps between poststroke days
        remapped_values = {value: index for index, value in enumerate(data_df['rel_day'].unique())}
        remapped_days = np.array([remapped_values[value] for value in data_df['rel_day']])
        data_df['x'] = remapped_days

        # For each day, get the fraction of in_RZ and out_RZ
        fracs = data_df.groupby('x').mean()
        fracs['out_rz'] = 1-fracs[var]
        rel_day = fracs.pop('rel_day')
        fracs.columns = ['in RZ', 'out RZ']

        # plt.figure()
        # axis = plt.subplot(111)
        axis = fracs.plot(kind='bar', stacked=True, ax=axis, width=1, color=['green', 'royalblue'], legend=False)
        axis.spines[['right', 'top']].set_visible(False)
        axis.set_xticklabels(axis.get_xticks(), rotation=0)
        axis.set(xticks=axis.get_xticks(), xticklabels=rel_day.astype(int), ylim=(0, 1),
                 ylabel='PFs in reward zones [%]', xlabel='')

        # Draw a horizontal line at chance level
        axis.axhline(total_zones, linestyle='--', c='black')
        axis.axvline(np.where(data_df['rel_day'].unique() > 0)[0][0]-0.5, linestyle='--', c='red')
        axis.set_title(f'M{m_id} - {num_cells} "Always-PC" cells')

        return axis

    attributes = {'com': plot_coms, 'pf_quad': plot_quadrants, 'pf_rz': plot_rzs, 'pf_zone': plot_zones}

    # Plot CoM location in corridor across sessions
    for a, func in attributes.items():
        fig, ax = plt.subplots(3, 3, sharey='all', layout='constrained')
        ax_flat = ax.flatten()
        if mouse_order is None:
            mouse_order = influence_df['mouse_id'].unique()
        for i, mouse in enumerate(mouse_order):
            func(data_df=influence_df[influence_df['mouse_id'] == mouse][['rel_day', a]], axis=ax_flat[i], m_id=mouse,
                 var=a, num_cells=influence_df[influence_df['mouse_id'] == mouse]['df_label_idx'].nunique())
        # ax_flat[-1].set_visible(False)

characterize_class3(influence_df=influence, rz_border_buffer=0, mouse_order=[41, 69, 85, 90, 33, 93, 121, 115, 122])
