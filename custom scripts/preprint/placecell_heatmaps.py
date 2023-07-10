#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 22/06/2023 11:33
@author: hheise

"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from schema import hheise_placecell, common_match, hheise_behav
from preprint import placecell_heatmap_transition_functions as func

# #%% Load matched/fetched data
# import pickle
# import os
# dir = r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Data\analysis\matched_data'
# with open(os.path.join(dir, 'match_matrices.pickle'), "r") as output_file:
#     match_matrices = pickle.load(output_file)
# with open(os.path.join(dir, 'is_pc.pickle'), "r") as output_file:
#     is_pc = pickle.load(output_file)
# with open(os.path.join(dir, 'pfs.pickle'), "r") as output_file:
#     pfs = pickle.load(output_file)
# with open(os.path.join(dir, 'spatial_maps.pickle'), "r") as output_file:
#     spatial_maps = pickle.load(output_file)
# with open(os.path.join(dir, 'spat_dff_maps.pickle'), "r") as output_file:
#     spat_dff_maps = pickle.load(output_file)

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
recovery = [33, 85, 86, 89, 90, 113]
no_recovery = [41, 63, 69, 110, 121]


#%%
# Deficit (recovery or no recovery) mice
queries = (
           # (common_match.MatchedIndex & 'mouse_id=33'),
           # (common_match.MatchedIndex & 'mouse_id=38'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=41'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=69' & 'day<="2021-03-23"'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=85'),
           (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=90'),
           # (common_match.MatchedIndex & 'mouse_id=93'),
           # (common_match.MatchedIndex & 'mouse_id=108' & 'day<"2022-09-09"'),
           # (common_match.MatchedIndex & 'mouse_id=110' & 'day<"2022-09-09"'),
           # (common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=121' & 'day<"2022-09-09"'),
           # (common_match.MatchedIndex & 'mouse_id=115' & 'day<"2022-09-09"'),
           # (common_match.MatchedIndex & 'mouse_id=122' & 'day<"2022-09-09"'),
           # (common_match.MatchedIndex & 'mouse_id=114' & 'day<"2022-09-09"')
)

# Sham mice
queries = (
           # (common_match.MatchedIndex & 'mouse_id=33' & 'day<="2020-08-24"'),
           (common_match.MatchedIndex & 'mouse_id=115' & 'day<"2022-09-09"'),
           (common_match.MatchedIndex & 'mouse_id=122' & 'day<"2022-09-09"'),
           (common_match.MatchedIndex & 'mouse_id=121' & 'day<"2022-09-09"'),
    # (common_match.MatchedIndex & 'mouse_id=114' & 'day<="2022-08-09"')
)

# All mice
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

is_pc = []
pfs = []
# spatial_maps = []
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

#%%
separate_days = True

if separate_days:
    # Stack together data from different mice (deficit)
    spat = []
    spat.append(func.get_place_cells(is_pc[0]['33_1'], spat_dff_maps[0]['33_1'], pc_day=-2, select_days=[-6, -5, -2, 1, 4, 13, 16]))
    spat.append(func.get_place_cells(is_pc_arr=is_pc[1]['41_1'], spat_arr=spat_dff_maps[1]['41_1'], pc_day=-1, select_days=[-5, -4, -1, 2, 5, 14, 17]))
    spat.append(func.get_place_cells(is_pc[2]['69_1'], spat_dff_maps[2]['69_1'], pc_day=0, select_days=[-2, -1, 0, 3, 6, 12, 15]))
    spat.append(func.get_place_cells(is_pc[3]['85_1'], spat_dff_maps[3]['85_1'], pc_day=0, select_days=[-2, -1, 0, 3, 5, 14, 17]))
    spat.append(func.get_place_cells(is_pc[4]['86_1'], spat_dff_maps[4]['86_1'], pc_day=0, select_days=[-2, -1, 0, 3, 5, 14, 17]))
    spat.append(func.get_place_cells(is_pc[5]['90_1'], spat_dff_maps[5]['90_1'], pc_day=0, select_days=[-2, -1, 0, 3, 5, 14, 17]))
    # spat_dff.append(get_place_cells(is_pc_arr=is_pc[4]['121_1'], spat_dff_arr=spat_dff_maps[4]['121_1'], pc_day=0, select_days=[-2, -1, 0, 3, 9, 18]))
    spat = np.vstack(spat)

    spat_noncoding = []
    spat_noncoding.append(func.get_noncoding(is_pc[0]['33_1'], spat_dff_maps[0]['33_1'], select_days=[-6, -5, -2, 1, 4, 13, 16]))
    spat_noncoding.append(func.get_noncoding(is_pc_arr=is_pc[1]['41_1'], spat_arr=spat_dff_maps[1]['41_1'], select_days=[-5, -4, -1, 2, 5, 14, 17]))
    spat_noncoding.append(func.get_noncoding(is_pc[2]['69_1'], spat_dff_maps[2]['69_1'], select_days=[-2, -1, 0, 3, 6, 12, 15]))
    spat_noncoding.append(func.get_noncoding(is_pc[3]['85_1'], spat_dff_maps[3]['85_1'], select_days=[-2, -1, 0, 3, 5, 14, 17]))
    spat_noncoding.append(func.get_noncoding(is_pc[4]['86_1'], spat_dff_maps[4]['86_1'], select_days=[-2, -1, 0, 3, 5, 14, 17]))
    spat_noncoding.append(func.get_noncoding(is_pc[5]['90_1'], spat_dff_maps[5]['90_1'], select_days=[-2, -1, 0, 3, 5, 14, 17]))
    # spat_dff_noncoding.append(get_noncoding(is_pc[4]['121_1'], spat_dff_maps[4]['121_1'], select_days=[-2, -1, 0, 3, 9, 18]))
    spat_noncoding, noncoding_masks = list(map(list, zip(*spat_noncoding)))
    spat_noncoding = np.vstack(spat_noncoding)

    # Stack together data from different mice (sham)
    spat = []
    spat.append(func.get_place_cells(is_pc_arr=is_pc[6]['91_1'], spat_arr=spat_dff_maps[6]['91_1'], pc_day=0, select_days=[-2, -1, 0, 3, 6, 12, 18]))
    # spat.append(func.get_place_cells(is_pc_arr=is_pc[7]['93_1'], spat_arr=spat_dff_maps[7]['93_1'], pc_day=0, select_days=[-2, -1, 0, 3, 6, 12, 18]))
    # spat.append(func.get_place_cells(is_pc_arr=is_pc[8]['108_1'], spat_arr=spat_dff_maps[8]['108_1'], pc_day=0, select_days=[-2, -1, 0, 3, 6, 15, 18]))
    # spat.append(func.get_place_cells(is_pc_arr=is_pc[9]['114_1'], spat_arr=spat_dff_maps[9]['114_1'], pc_day=0, select_days=[-2, -1, 0, 3, 6, 12, 18]))
    spat.append(func.get_place_cells(is_pc_arr=is_pc[10]['115_1'], spat_arr=spat_dff_maps[10]['115_1'], pc_day=0, select_days=[-2, -1, 0, 3, 6, 15, 18]))
    spat.append(func.get_place_cells(is_pc_arr=is_pc[11]['122_1'], spat_arr=spat_dff_maps[11]['122_1'], pc_day=0, select_days=[-2, -1, 0, 3, 6, 15, 18]))
    spat = np.vstack(spat)

    spat_noncoding = []
    spat_noncoding.append(func.get_noncoding(is_pc_arr=is_pc[6]['91_1'], spat_arr=spat_dff_maps[6]['91_1'], select_days=[-2, -1, 0, 3, 6, 12, 18]))
    # spat_noncoding.append(func.get_noncoding(is_pc_arr=is_pc[7]['93_1'], spat_arr=spat_dff_maps[7]['93_1'], select_days=[-2, -1, 0, 3, 6, 12, 18]))
    # spat_noncoding.append(func.get_noncoding(is_pc_arr=is_pc[8]['108_1'], spat_arr=spat_dff_maps[8]['108_1'], select_days=[-2, -1, 0, 3, 6, 15, 18]))
    # spat_noncoding.append(func.get_noncoding(is_pc_arr=is_pc[9]['114_1'], spat_arr=spat_dff_maps[9]['114_1'], select_days=[-2, -1, 0, 3, 6, 12, 18]))
    spat_noncoding.append(func.get_noncoding(is_pc_arr=is_pc[10]['115_1'], spat_arr=spat_dff_maps[10]['115_1'], select_days=[-2, -1, 0, 3, 6, 15, 18]))
    spat_noncoding.append(func.get_noncoding(is_pc_arr=is_pc[11]['122_1'], spat_arr=spat_dff_maps[11]['122_1'], select_days=[-2, -1, 0, 3, 6, 15, 18]))
    spat_noncoding, noncoding_masks = list(map(list, zip(*spat_noncoding)))
    spat_noncoding = np.vstack(spat_noncoding)

else:
    # AVERAGE DIFFERENT PERIODS
    first_poststroke_separate = True    # Whether the first poststroke session should be shown separately or included in early phase
    pc_fraction = 0.2   # Fraction of sessions that a cell has to be a PC to be included

    # Stack together data from different mice (deficit)
    spat = []
    spat.append(func.get_place_cells_multiday(is_pc_arr=is_pc[0]['41_1'], spat_arr=spat_dff_maps[0]['41_1'], pc_frac=pc_fraction, first_post=first_poststroke_separate))
    spat.append(func.get_place_cells_multiday(is_pc[1]['69_1'], spat_dff_maps[1]['69_1'], pc_frac=pc_fraction, first_post=first_poststroke_separate))
    spat.append(func.get_place_cells_multiday(is_pc[2]['85_1'], spat_dff_maps[2]['85_1'], pc_frac=pc_fraction, first_post=first_poststroke_separate))
    spat.append(func.get_place_cells_multiday(is_pc[3]['90_1'], spat_dff_maps[3]['90_1'], pc_frac=pc_fraction, first_post=first_poststroke_separate))
    # spat.append(get_place_cells_multiday(is_pc_arr=is_pc[4]['121_1'], spat_arr=spat_dff_maps[4]['121_1'], pc_day=0, select_days=[-2, -1, 0, 3, 9, 18]))
    spat, pc_stab = list(map(list, zip(*spat)))
    spat = np.vstack(spat)
    pc_stab = np.concatenate(pc_stab)

    spat_noncoding = []
    spat_noncoding.append(func.get_noncoding_multiday(is_pc[0]['41_1'], spat_dff_maps[0]['41_1'], first_post=first_poststroke_separate))
    spat_noncoding.append(func.get_noncoding_multiday(is_pc[1]['69_1'], spat_dff_maps[1]['69_1'], first_post=first_poststroke_separate))
    spat_noncoding.append(func.get_noncoding_multiday(is_pc[2]['85_1'], spat_dff_maps[2]['85_1'], first_post=first_poststroke_separate))
    spat_noncoding.append(func.get_noncoding_multiday(is_pc[3]['90_1'], spat_dff_maps[3]['90_1'], first_post=first_poststroke_separate))
    # spat_dff_noncoding.append(get_noncoding(is_pc[4]['121_1'], spat_dff_maps[4]['121_1'], select_days=[-2, -1, 0, 3, 9, 18]))
    noncoding_masks, spat_noncoding = list(map(list, zip(*spat_noncoding)))
    spat_noncoding = np.vstack(spat_noncoding)

    # Stack together data from different mice (sham)
    spat = []
    spat.append(func.get_place_cells_multiday(is_pc_arr=is_pc[4]['115_1'], spat_arr=spat_dff_maps[4]['115_1'], pc_frac=pc_fraction, first_post=first_poststroke_separate))
    spat.append(func.get_place_cells_multiday(is_pc_arr=is_pc[5]['122_1'], spat_arr=spat_dff_maps[5]['122_1'], pc_frac=pc_fraction, first_post=first_poststroke_separate))
    spat.append(func.get_place_cells_multiday(is_pc_arr=is_pc[6]['121_1'], spat_arr=spat_dff_maps[6]['121_1'], pc_frac=pc_fraction, first_post=first_poststroke_separate))
    spat, pc_stab = list(map(list, zip(*spat)))
    spat = np.vstack(spat)
    pc_stab = np.concatenate(pc_stab)

    spat_noncoding = []
    spat_noncoding.append(func.get_noncoding_multiday(is_pc[4]['115_1'], spat_dff_maps[4]['115_1'], first_post=first_poststroke_separate))
    spat_noncoding.append(func.get_noncoding_multiday(is_pc[5]['122_1'], spat_dff_maps[5]['122_1'], first_post=first_poststroke_separate))
    spat_noncoding.append(func.get_noncoding_multiday(is_pc[6]['121_1'], spat_dff_maps[6]['121_1'], first_post=first_poststroke_separate))
    noncoding_masks, spat_noncoding = list(map(list, zip(*spat_noncoding)))
    spat_noncoding = np.vstack(spat_noncoding)


# split dataset into stable (upper 50%) and unstable (lower 50%) cells
upper_percentile = 50
lower_percentile = 50

if separate_days:
    pc_stab = func.get_stab(spat)
    upper_thresh = np.percentile(pc_stab, upper_percentile)
    lower_thresh = np.percentile(pc_stab, lower_percentile)
else:
    upper_thresh = np.nanpercentile(pc_stab, upper_percentile)
    lower_thresh = np.nanpercentile(pc_stab, lower_percentile)

spat_stable = spat[pc_stab > upper_thresh]
spat_unstable = spat[pc_stab <= lower_thresh]
spat_unstable = np.concatenate((spat_unstable, spat[np.isnan(pc_stab)]))    # Add cells with only 1 prestroke session (no stability) to unstable

# Sort by maximum activity location
spat_stable_sort = func.sort_neurons(spat_stable, day_idx=2)
spat_unstable_sort = func.sort_neurons(spat_unstable, day_idx=2)
spat_noncoding_sort = func.sort_neurons(spat_noncoding, day_idx=2)

# sort_noncoding_neurons(spat_dff_arr=spat_dff_noncoding, is_pc_list=is_pc, noncoding_list=noncoding_masks)

# SHAM: Filter out some bad neurons
spat_noncoding_sort = np.delete(spat_noncoding_sort, [170], axis=0)
func.draw_heatmap_across_days(data_arrays=[spat_stable_sort, spat_unstable_sort, spat_noncoding_sort],
                         titles=[-2, -1, 0, 3, 6, 15, 18], draw_empty_row=True, draw_zone_borders=True)


sham = ['91_1', '115_1', ' 122_1']                  # no deficit, and < 50 spheres
no_deficit = ['83_1', '95_1', '112_1', '114_1', '116_1']   # no deficit, but > 50 spheres
late_deficit = ['111_1', '93_1', '108_1']                          # Debatable, might be merged with sham
recovery = ['33_1', '85_1', '86_1', '90_1', '113_1']
no_recovery = ['41_1', '63_1', '69_1', '89_1', '110_1']
no_recovery_with121 = ['41_1', '63_1', '69_1', '89_1', '110_1', '121_1']

control = ['91_1', '93_1', '108_1', '114_1', '115_1', '122_1']
stroke = ['33_1', '41_1', '69_1', '85_1', '86_1', '90_1']
stroke_with121 = ['33_1', '41_1', '69_1', '85_1', '86_1', '90_1', '121_1']

groups = {'sham': sham, 'no_deficit': no_deficit, 'late_deficit': late_deficit, 'recovery': recovery,
          'no_recovery': no_recovery, 'no_recovery_with121': no_recovery_with121, 'control': control,
          'stroke': stroke, 'stroke_with121': stroke_with121}

mice = ['33_1', '41_1', '69_1', '85_1', '86_1', '90_1', '91_1', '93_1', '108_1', '114_1', '115_1', '122_1', '121_1']
dfs = []
for i, mouse in enumerate(mice):
    rel_dates = np.array(is_pc[i][mouse][1])
    mask_pre = rel_dates <= 0
    mask_early = (0 < rel_dates) & (rel_dates <= 6)
    mask_late = rel_dates > 6

    classes_pre, stability_thresh = func.quantify_stability_split(is_pc_arr=is_pc[i][mouse][0][:, mask_pre],
                                                             spat_dff_arr=spat_dff_maps[i][mouse][0][:, mask_pre],
                                                             rel_days=rel_dates[mask_pre], mouse_id=mouse)
    classes_early = func.quantify_stability_split(is_pc_arr=is_pc[i][mouse][0][:, mask_early],
                                             spat_dff_arr=spat_dff_maps[i][mouse][0][:, mask_early],
                                             rel_days=rel_dates[mask_early], stab_thresh=stability_thresh)
    classes_late = func.quantify_stability_split(is_pc_arr=is_pc[i][mouse][0][:, mask_late],
                                            spat_dff_arr=spat_dff_maps[i][mouse][0][:, mask_late],
                                            rel_days=rel_dates[mask_late], stab_thresh=stability_thresh)

    df = pd.concat([func.summarize_quantification(classes_pre, 'pre'),
                    func.summarize_quantification(classes_early, 'early'),
                    func.summarize_quantification(classes_late, 'late')])
    df['mouse_id'] = mouse
    df['classes'] = [classes_pre, classes_early, classes_late]
    dfs.append(df)
class_df = pd.concat(dfs, ignore_index=True)

class_df_prism = func.pivot_classdf_prism(class_df, percent=True, col_order=['pre', 'early', 'late'])
class_df_prism.to_clipboard()

# Plot transition matrices
def label_row(label, axis, pad=5):
    axis.annotate(label, xy=(0, 0.5), xytext=(-axis.yaxis.labelpad - pad, 0), xycoords=axis.yaxis.label,
                  textcoords='offset points', size='large', ha='right', va='center')


fig, ax = plt.subplots(2, len(class_df.mouse_id.unique()), layout='constrained')
matrices = []
for i, mouse in enumerate(class_df.mouse_id.unique()):
    mask_pre = class_df[(class_df['mouse_id'] == mouse) & (class_df['period'] == 'pre')]['classes'].iloc[0]
    mask_early = class_df[(class_df['mouse_id'] == mouse) & (class_df['period'] == 'early')]['classes'].iloc[0]
    mask_late = class_df[(class_df['mouse_id'] == mouse) & (class_df['period'] == 'late')]['classes'].iloc[0]

    pre_early_trans = func.transition_matrix(mask_pre, mask_early, percent=False)
    early_late_trans = func.transition_matrix(mask_early, mask_late, percent=False)

    # sns.heatmap(pre_early_trans, ax=ax[0, i], square=True, annot=True, cbar=False)
    # sns.heatmap(early_late_trans, ax=ax[1, i], square=True, annot=True, cbar=False)
    # ax[0, i].set_title(mouse)

    matrices.append(pd.DataFrame([dict(mouse_id=mouse, pre_early=pre_early_trans, early_late=early_late_trans)]))
matrices = pd.concat(matrices, ignore_index=True)

# label_row('Pre->Early', ax[0, 0])
# label_row('Early->Late', ax[1, 0])
ax[0, 0].set_xlabel('To Cell Class Early')
ax[0, 0].set_ylabel('From Cell Class Pre')
ax[1, 0].set_xlabel('To Cell Class Late')
ax[1, 0].set_ylabel('From Cell Class Early')

# Make average transition matrices
avg_trans = {}
for k, v in groups.items():
    avg_trans[k+'_early'] = np.mean(np.array(list(matrices[matrices['mouse_id'].isin(v)]['pre_early'])), axis=0)
    avg_trans[k+'_late'] = np.mean(np.array(list(matrices[matrices['mouse_id'].isin(v)]['early_late'])), axis=0)

fig, ax = plt.subplots(2, 4, layout='constrained')
titles = ['deficit all', 'no recovery', 'recovery', 'sham']
for i, mat in enumerate([[avg_trans['deficit_early'], avg_trans['deficit_late']],
                         [avg_trans['no_recovery_early'], avg_trans['no_recovery_late']],
                         [avg_trans['recovery_early'], avg_trans['recovery_late']],
                         [avg_trans['sham_early'], avg_trans['sham_late']]]):
    sns.heatmap(mat[0], ax=ax[0, i], square=True, annot=True, cbar=False)
    sns.heatmap(mat[1], ax=ax[1, i], square=True, annot=True, cbar=False)
    ax[0, i].set_title(titles[i])
ax[0, 0].set_xlabel('To Cell Class Early')
ax[0, 0].set_ylabel('From Cell Class Pre')
ax[1, 0].set_xlabel('To Cell Class Late')
ax[1, 0].set_ylabel('From Cell Class Early')


### QUANTIFY TRANSITION MATRICES WITH STATISTICAL TESTS ###
from scipy import stats
stroke_early = np.array(list(matrices[matrices['mouse_id'].isin(stroke)]['pre_early']))
stroke_late = np.array(list(matrices[matrices['mouse_id'].isin(stroke)]['early_late']))
no_deficit_early = np.array(list(matrices[matrices['mouse_id'].isin(no_deficit)]['pre_early']))
no_deficit_late = np.array(list(matrices[matrices['mouse_id'].isin(no_deficit)]['early_late']))

early_ttest = func.transition_matrix_ttest(stroke_early, no_deficit_early)
late_ttest = func.transition_matrix_ttest(stroke_late, no_deficit_late)

export_early = func.export_trans_matrix(stroke_early, no_deficit_early)
export_late = func.export_trans_matrix(stroke_late, no_deficit_late)

### Transition heatmaps
# Cell numbers in early and late dont sum up because there are some cells that transitioned from noncoding to PC
def_heat_early = func.transition_heatmap(classes=class_df[class_df['mouse_id'].isin(deficit)], spat_arr=spat_dff_maps,
                                    to_period='early', place_cells=False)
def_heat_late = func.transition_heatmap(classes=class_df[class_df['mouse_id'].isin(deficit)], spat_arr=spat_dff_maps,
                                   to_period='late', place_cells=False)

sham_heat_early = func.transition_heatmap(classes=class_df[class_df['mouse_id'].isin(sham)], spat_arr=spat_dff_maps,
                                     to_period='early', place_cells=False)
sham_heat_late = func.transition_heatmap(classes=class_df[class_df['mouse_id'].isin(sham)], spat_arr=spat_dff_maps,
                                    to_period='late', place_cells=False)

# Stroke noncoding prestroke transition heatmaps have some outliers
def_heat_early['from_act'] = np.delete(def_heat_early['from_act'], [394, 424, 425, 442, 372], axis=0)

## Plot heatmaps
func.plot_transition_heatmaps(early_maps=def_heat_early, late_maps=def_heat_late)     # Deficit place cells
func.plot_transition_heatmaps(early_maps=sham_heat_early, late_maps=sham_heat_late)     # Sham place cells

#%% Transition matrix visualizations
### Visualize transition matrices with hmmviz
from hmmviz import TransGraph

# Use pandas for cross-tabulation (transition matrix)
classes = ['lost', 'non-coding', 'unstable', 'stable']
deficit_early_tab = pd.DataFrame(deficit_early, columns=classes, index=classes)
deficit_late_tab = pd.DataFrame(deficit_late, columns=classes, index=classes)
no_recovery_early_tab = pd.DataFrame(no_recovery_early, columns=classes, index=classes)
no_recovery_late_tab = pd.DataFrame(no_recovery_late, columns=classes, index=classes)
recovery_early_tab = pd.DataFrame(recovery_early, columns=classes, index=classes)
recovery_late_tab = pd.DataFrame(recovery_late, columns=classes, index=classes)
sham_early_tab = pd.DataFrame(sham_early, columns=classes, index=classes)
sham_late_tab = pd.DataFrame(sham_late, columns=classes, index=classes)

graph = TransGraph(deficit_early_tab/100)
fig = plt.figure(figsize=(12, 12))
graph.draw(edgelabels=True)

# Export classes for Plotly Sankey Diagram
matrices = []
for i, mouse in enumerate(class_df.mouse_id.unique()):
    mask_pre = class_df[(class_df['mouse_id'] == mouse) & (class_df['period'] == 'pre')]['classes'].iloc[0]
    mask_early = class_df[(class_df['mouse_id'] == mouse) & (class_df['period'] == 'early')]['classes'].iloc[0]
    mask_late = class_df[(class_df['mouse_id'] == mouse) & (class_df['period'] == 'late')]['classes'].iloc[0]

    pre_early_trans = func.transition_matrix(mask_pre, mask_early, percent=False)
    early_late_trans = func.transition_matrix(mask_early, mask_late, percent=False)

    matrices.append(pd.DataFrame([dict(mouse_id=mouse, pre_early=pre_early_trans, early_late=early_late_trans)]))
matrices = pd.concat(matrices, ignore_index=True)

# Make average transition matrices
avg_trans = {}
for k, v in groups.items():
    avg_trans[k+'_early'] = np.mean(np.array(list(matrices[matrices['mouse_id'].isin(v)]['pre_early'])), axis=0)
    avg_trans[k+'_late'] = np.mean(np.array(list(matrices[matrices['mouse_id'].isin(v)]['early_late'])), axis=0)


def unravel_matrix(mat_early: np.ndarray, mat_late: np.ndarray):
    source_early = [it for sl in [[i]*4 for i in range(4)] for it in sl]  # "From" class
    target_early = [4, 5, 6, 7]*4                      # "To" classes (shifted by 4)
    source_late = np.array(source_early) + 4
    target_late = np.array(target_early) + 4        # Classes from early->late have different labels (shifted by 4 again)

    early_flat = mat_early.flatten()
    late_flat = mat_late.flatten()

    # Concatenate everything into three rows (source, target, value)
    out = np.array([[*source_early, *source_late], [*target_early, *target_late], [*early_flat, *late_flat]])
    return out


for k in groups.keys():
    np.savetxt(os.path.join(r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Papers\preprint\pc_heatmaps\sankey_plot',
                            f'{k}.csv'), unravel_matrix(avg_trans[k+'_early'], avg_trans[k+'_late']), fmt="%d", delimiter=',')
