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

from schema import common_match, common_img, hheise_placecell, hheise_behav, common_mice, hheise_hist
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
cat_cm = matplotlib.colors.ListedColormap(np.array([[0.5, 0.5, 0.5, 1],     # Class 0: brown
                                                    [0.0, 0.6, 1.0, 1],     # Class 0: blue
                                                    [0.2, 0.8, 0.2, 1],     # Class 0: green
                                                    [1.0, 0.4, 0.0, 1]]))   # Class 0: orange

fig, ax = plt.subplots(2, 3)
ax[0, 0] = plot_colored_cells(match_matrix=match_matrices[0]['41_1'], mouse_id=41, day='2020-08-27', axis=ax[0, 0],
                              row_ids=np.arange(len(pc_fractions[41])), color=pc_fractions[41], title='41', cmap=cat_cm,
                              draw_cbar=False, background=None)
ax[0, 1] = plot_colored_cells(match_matrix=match_matrices[1]['69_1'], mouse_id=69, day='2021-03-11', axis=ax[0, 1],
                              row_ids=np.arange(len(pc_fractions[69])), color=pc_fractions[69], title='69', cmap=cat_cm,
                              cbar_ticklabels=['Never PC', 'PC post only', 'PC pre only', 'Always PC'], background=None)
ax[0, 2].set_visible(False)
ax[1, 0] = plot_colored_cells(match_matrix=match_matrices[2]['121_1'], mouse_id=121, day='2022-08-15', axis=ax[1, 0],
                              row_ids=np.arange(len(pc_fractions[121])), color=pc_fractions[121], title='121',
                              cmap=cat_cm, draw_cbar=False, background=None)
ax[1, 1] = plot_colored_cells(match_matrix=match_matrices[3]['115_1'], mouse_id=115, day='2022-08-12', axis=ax[1, 1],
                              row_ids=np.arange(len(pc_fractions[115])), color=pc_fractions[115], title='115',
                              cmap=cat_cm, draw_cbar=False, background=None)
ax[1, 2] = plot_colored_cells(match_matrix=match_matrices[4]['122_1'], mouse_id=122, day='2022-08-15', axis=ax[1, 2],
                              row_ids=np.arange(len(pc_fractions[122])), color=pc_fractions[122], title='122',
                              cmap=cat_cm, draw_cbar=False, background=None)


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

# Classes of possible PF locations
columns = np.array(['pre_RZ1', 'in_RZ1', 'RZ1-RZ2', 'in_RZ2', 'RZ2-RZ3', 'in_RZ3', 'RZ3-RZ4', 'in_RZ4', 'post_RZ4'])
rz_mask = np.array([False, True, False, True, False, True, False, True, False])

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
        summary = {col: np.nan for col in columns}
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
        pf[columns[idx]] = pf['com'].between(prev_border, border)
        prev_border = border
    pf[columns[idx+1]] = pf['com'].between(prev_border, 80)

    # Add columns for in vs outside of RZs
    pf['in_RZ'] = pf[columns[rz_mask]].any(axis=1)
    pf['out_RZ'] = pf[columns[~rz_mask]].any(axis=1)

    # Make summaries
    n_cells = pf['mask_id'].nunique()
    summary = {col: pf[col].sum()/n_cells for col in columns}
    summary['in_RZ'] = pf['in_RZ'].sum()/n_cells
    summary['out_RZ'] = pf['out_RZ'].sum() / n_cells
    pc_location.append(pd.DataFrame(data=[dict(mouse_id=pk['mouse_id'], day=pk['day'], rel_day=rel_day, **summary)]))

    summary = {col: pf[col].sum()/len(pf) for col in columns}
    summary['in_RZ'] = pf['in_RZ'].sum()/len(pf)
    summary['out_RZ'] = pf['out_RZ'].sum() / len(pf)
    pf_location.append(pd.DataFrame(data=[dict(mouse_id=pk['mouse_id'], day=pk['day'], rel_day=rel_day, **summary)]))

pf_loc = pd.concat(pf_location)
pc_loc = pd.concat(pc_location)

# Export in/out data for line plots in Prism
pf_loc_export = pf_loc.pivot(index='rel_day', columns='mouse_id', values='in_RZ')
pf_loc_export.to_csv(r'C:\Users\hheise.UZH\Desktop\pc_location\pf_loc.csv')
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

def measure_cell_influence(cell_class, pf_center, mouse_id, match_matrix, days, exclude_other_class3_cells=False):

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

    print(mouse_id)

    # Only take accepted PFs from accepted PCs
    pf_k = dict(place_cell_id=2, corridor_type=0, is_place_cell=1, large_enough=1, strong_enough=1, transients=1)
    bord = (hheise_behav.CorridorPattern() & 'pattern="training"').rescale_borders(80)

    # For all class-3 cells, compute fraction of other place cells that are similar/average distance of place fields
    # all sessions where class-3 cells are actually place cells
    class3 = np.where(cell_class == 3)[0]
    class3_df = []
    for day, relative_day in zip(pf_center.columns, days):
        # If there are class-3-cells in this session, start loading data for all place cells
        if not pf_center[day].iloc[class3].isna().all():
            real_class3_ids = match_matrix[day].iloc[class3].values
            all_curr_pfs = pd.DataFrame((hheise_placecell.PlaceCell.ROI * hheise_placecell.PlaceCell.PlaceField &
                                         f'mouse_id={mouse_id}' & pf_k &
                                         common_match.MatchedIndex().string2key(title=day)).fetch('KEY', 'bin_idx', as_dict=True))

            # Compute PF CoM for all place cells of the current session
            dff_pks, all_spat_dff = (hheise_placecell.BinnedActivity.ROI & all_curr_pfs).get_normal_act(trace='dff', return_pks=True)
            all_spat = pd.DataFrame(dff_pks)
            all_spat['spat_dff'] = list(all_spat_dff)
            all_pc = pd.merge(all_curr_pfs, all_spat)
            all_pf_com = [dc.place_field_com(dff, bin_idx)[0] for bin_idx, dff in zip(all_pc['bin_idx'], all_pc['spat_dff'])]
            all_pc['com'] = all_pf_com
            # Make a dummy 5th border (same distance as RZ1-RZ2) for place fields after the 4th RZ
            all_pc['pf_quad'] = all_pc['com'].apply(get_distance, borders=np.append(bord[:, 0], bord[-1, 0]+(bord[1, 0]-bord[0, 0])))
            all_pc['pf_rz'] = all_pc['com'].apply(in_reward_zone, borders=bord)
            all_pc['pf_zone'] = all_pc['com'].apply(get_zone, borders=bord)

            if exclude_other_class3_cells:
                all_class3_cells = all_pc[all_pc['mask_id'].isin(real_class3_ids)]
                all_pc = all_pc[~all_pc['mask_id'].isin(real_class3_ids)]

            for df_idx, dj_mask_id in zip(class3, real_class3_ids):
                # for each class3 cell, compute distance/similarity of the c3 cell to all other PCs (also non-tracked) at that session

                # TODO: M69, day 20210303, row 69, mask_id 254 is not in all_pc, but in pf_center
                #  also, for M69, day 20210303, row 31, mask_id 154, pf_center has different com as all_pc

                if exclude_other_class3_cells:
                    # If we excluded all class3 cells before, we dont have to do it for each cell
                    curr_class3 = all_class3_cells[all_class3_cells['mask_id'] == dj_mask_id].squeeze().to_dict()
                    rest_pc = all_pc
                else:
                    # Otherwise, remove the current class 3 cell from the DF for processing
                    curr_class3 = all_pc[all_pc['mask_id'] == dj_mask_id].squeeze().to_dict()
                    rest_pc = all_pc[all_pc['mask_id'] != dj_mask_id]

                class3_df.append(pd.DataFrame([dict(match_matrix_idx=df_idx, day=day, rel_day=relative_day, mask_idx=int(dj_mask_id), n_rest_pc=len(rest_pc),
                                  com=curr_class3['com'], pf_quad=curr_class3['pf_quad'], pf_rz=curr_class3['pf_rz'], pf_zone=curr_class3['pf_zone'],
                                  avg_dist=(rest_pc['com'] - curr_class3['com']).abs().mean() * (400/80),               # Average distance of PF CoM to other accepted PFs in cm
                                  avg_quad_dist=(rest_pc['pf_quad'] - curr_class3['pf_quad']).abs().mean() * (400/80),  # Average distance of PF CoM to other accepted PFs, normalized by quadrants, in cm
                                  share_rz=(rest_pc['pf_rz'] == curr_class3['pf_rz']).sum()/len(rest_pc),      # Fraction of PFs that are also in a reward zone/not in reward zone
                                  share_zone=(rest_pc['pf_zone'] == curr_class3['pf_zone']).sum()/len(rest_pc),
                                  )]))
    class3_df = pd.concat(class3_df)
    return class3_df


# 1.
pc_fractions = {41: compute_placecell_fractions(pc=is_pc[0]['41_1'][0], days=np.array(is_pc[0]['41_1'][1])),
                69: compute_placecell_fractions(pc=is_pc[1]['69_1'][0], days=np.array(is_pc[1]['69_1'][1])),
                121: compute_placecell_fractions(pc=is_pc[2]['121_1'][0], days=np.array(is_pc[2]['121_1'][1])),
                115: compute_placecell_fractions(pc=is_pc[3]['115_1'][0], days=np.array(is_pc[3]['115_1'][1])),
                122: compute_placecell_fractions(pc=is_pc[4]['122_1'][0], days=np.array(is_pc[4]['122_1'][1]))}

# 2.
pf_centers = {41: get_pf_location(match_matrix=match_matrices[0]['41_1'], spatial_dff=spatial_maps[0]['41_1'][0], place_fields=pfs[0]['41_1'][0]),
              69: get_pf_location(match_matrix=match_matrices[1]['69_1'], spatial_dff=spatial_maps[1]['69_1'][0], place_fields=pfs[1]['69_1'][0]),
              121: get_pf_location(match_matrix=match_matrices[2]['121_1'], spatial_dff=spatial_maps[2]['121_1'][0], place_fields=pfs[2]['121_1'][0]),
              115: get_pf_location(match_matrix=match_matrices[3]['115_1'], spatial_dff=spatial_maps[3]['115_1'][0], place_fields=pfs[3]['115_1'][0]),
              122: get_pf_location(match_matrix=match_matrices[4]['122_1'], spatial_dff=spatial_maps[4]['122_1'][0], place_fields=pfs[4]['122_1'][0])}

# 3.
influence = {k: measure_cell_influence(cell_class=pc_fractions[k], pf_center=pf_centers[k], mouse_id=k,
                                       match_matrix=match_matrices[i][f'{k}_1'], days=pfs[i][f'{k}_1'][1])
             for i, k in enumerate(pf_centers)}
