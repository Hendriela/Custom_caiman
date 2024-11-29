#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 20/11/2024 17:14
@author: hheise

Code for new analysis and requests that came during the first revision round in November 2024.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from schema import (hheise_grouping, hheise_placecell, common_mice, common_match, hheise_behav,  hheise_hist,
                    common_hist, hheise_connectivity)
from util import helper

####
# Create table with basic mouse information
include_mice = [33, 41, 63, 69, 83, 85, 86, 89, 90, 91, 93, 95, 108, 110, 111, 113, 114, 115, 116, 122]
df = pd.DataFrame((common_mice.Mouse *
                   hheise_placecell.TransientOnly.ROI *
                   hheise_grouping.BehaviorGrouping &
                   f'mouse_id in {helper.in_query(include_mice)}' &
                   'cluster = "fine"' & 'grouping_id = 4' &
                   'place_cell_id=2' & 'day <= "2022-09-09"'
                   ).fetch('mouse_id', 'strain', 'sex', 'group', 'day', as_dict=True))
df_counts = df.groupby(by=['mouse_id', 'strain', 'sex', 'group', 'day']).size().reset_index(name='counts')

n_tracked_cells = []
for mouse in include_mice:
    if mouse == 63:
        start_with_ref = True
    else:
        start_with_ref = False
    mat = (common_match.MatchedIndex() & 'username="hheise"' & f'mouse_id={mouse}').construct_matrix(start_with_ref=start_with_ref)
    n_tracked_cells.append(dict(mouse_id=mouse, num_tracked_cells=len(mat[list(mat.keys())[0]])))

df_merge = pd.merge(left=df_counts, right=pd.DataFrame(n_tracked_cells), on='mouse_id', how='left')
df_mice = df_counts.groupby(['mouse_id', 'strain', 'sex', 'group'])['counts'].apply(list).reset_index(name='count_list')
df_mice.to_clipboard()


####
# Plot lick histograms of anticipatory licking
sample_sessions = ['2021-02-16', '2021-03-04', '2021-03-11', '2021-03-23']
(hheise_behav.VRSession & 'mouse_id=69' & f'day in {helper.in_query(sample_sessions)}').plot_lick_histogram(metrics=['binned_lick_ratio', 'si_binned_run'])

sample_sessions = ['2021-02-14', '2021-03-02', '2021-03-06', '2021-03-17']
(hheise_behav.VRSession & 'mouse_id=63' & f'day in {helper.in_query(sample_sessions)}').plot_lick_histogram(metrics=['binned_lick_ratio', 'si_binned_run'])

####
# Plot average stroke volume per microsphere/infarct for recovery, no-recovery, sham

# Get lesion volumes of each individual infarct for all brain regions
df = pd.DataFrame((hheise_hist.Microsphere() * common_hist.Histology & 'username="hheise"' & 'lesion=1' & 'mouse_id!=94').fetch('mouse_id', 'thickness', 'auto', as_dict=True))
df['vol'] = df['auto'] * (df['thickness'] / 1000)
fine = (hheise_grouping.BehaviorGrouping & 'grouping_id = 4' & 'cluster = "fine"').get_groups()
df = df.merge(fine, how='left', on='mouse_id')
(df[(df.group == 'Recovery') & (df.auto > 0)]['auto'] * 1000).to_clipboard(index=False, header=False)
df['est_vol2'] = df['auto'] * (40 / 1000)
(df[(df.group == 'Recovery') & (df.est_vol2 > 0)]['est_vol2'] * 1000).to_clipboard(index=False, header=False)
df.groupby('mouse_id').agg({'est_vol': 'mean'}) * 1000

df = hheise_hist.Microsphere().get_structure_groups(grouping={'hippocampus': ['HPF'],
                                                              'white_matter': ['fiber tracts'],
                                                              'striatum': ['STR'],
                                                              'thalamus': ['TH'],
                                                              'neocortex': ['Isocortex'],
                                                              },
                                                    columns=['spheres', 'spheres_rel', 'spheres_lesion', 'spheres_lesion_rel', 'lesion_vol', 'lesion_vol_rel',
           'spheres_extrap', 'spheres_rel_extrap', 'spheres_lesion_extrap', 'spheres_lesion_rel_extrap',
           'region', 'mouse_id', 'username'], lesion_stains=['map2', 'gfap', 'auto'])


#####
# Plot distributions (violin plots) of neuron-neuron correlations across phases, groups and nc-nc vs pc-pc pairs

# Fetch correlation matrices per day
df = pd.DataFrame((hheise_connectivity.NeuronNeuronCorrelation * hheise_grouping.BehaviorGrouping * hheise_behav.RelDay & 'cluster="fine"' & 'grouping_id=4'
                  & 'place_cell_id=2' & 'trace_type="dff"').fetch('mouse_id', 'group', 'day', 'phase', 'rel_day_align', 'corr_matrix', 'pc_mask', as_dict=True))
df.loc[df.group == 'No Deficit', 'group'] = 'Sham'

# For each mouse and day, mask the correlation matrix with the pc_mask and store a flat array for nc-nc pairs and pc-pc pairs
def get_flat_pair_arrays(row):
    pc_mask = row['pc_mask'].astype(bool)
    corr_mat = row['corr_matrix']
    mat_pc = corr_mat[pc_mask, :][:, pc_mask].flatten()
    mat_nc = row['corr_matrix'][~pc_mask, :][:, ~pc_mask].flatten()

    return dict(mat_pc=mat_pc[~np.isnan(mat_pc.flatten())], mat_nc=mat_nc[~np.isnan(mat_nc.flatten())])


df_flat = df.join(pd.json_normalize(df.apply(get_flat_pair_arrays, axis=1)))

# Concatenate flat arrays across group and phase
new_df = []
for group_df in df_flat.groupby(['group', 'phase']):
    corr_pc = np.concatenate(group_df[1]['mat_pc'].to_list())
    corr_nc = np.concatenate(group_df[1]['mat_nc'].to_list())

    new_df.append(pd.DataFrame(dict(group=group_df[0][0],
                                    phase=group_df[0][1],
                                    corr=np.concatenate([corr_pc, corr_nc]),
                                    corr_label=[*['pc']*len(corr_pc), *['nc']*len(corr_nc)])))
new_df = pd.concat(new_df)
del df, df_flat

# Plot data across groups and function for each phase
fig, ax = plt.subplots(nrows=1, ncols=3, sharey='all', sharex='all', layout='constrained')
sns.violinplot(data=new_df[new_df.phase == 'pre'], x='group', y='corr', hue='corr_label', ax=ax[0],
               order=['Sham', 'Recovery', 'No Recovery'], hue_order=['nc', 'pc'], cut=0)
sns.violinplot(data=new_df[new_df.phase == 'early'], x='group', y='corr', hue='corr_label', ax=ax[1],
               order=['Sham', 'Recovery', 'No Recovery'], hue_order=['nc', 'pc'], cut=0)
sns.violinplot(data=new_df[new_df.phase == 'late'], x='group', y='corr', hue='corr_label', ax=ax[2],
               order=['Sham', 'Recovery', 'No Recovery'], hue_order=['nc', 'pc'], cut=0)

# Plot a CDF with all cells of prestroke sessions
sns.histplot(data=new_df[new_df.phase == 'pre'], x='corr', hue='corr_label', bins=500, binrange=(-1, 1),
             cumulative=True, common_norm=False, stat='percent', element="step", fill=False)

# CDFs work in visualizing, draw them for each period and group in a facetgrid
g = sns.displot(
    data=new_df, x="corr", col="group", row="phase", hue='corr_label',
    bins=300, cumulative=True, common_norm=False, stat='percent', element="step", fill=False,
    col_order=['Sham', 'Recovery', 'No Recovery'], row_order=['pre', 'early', 'late'], facet_kws=dict(margin_titles=True),
)

# Mark the 95th percentile
# flatten axes into a 1-d array
axes = g.axes.flatten()

# iterate through the axes
for i, ax in enumerate(axes):
    ax.axhline(95, ls='--', c='red')


######
# Relationship of distance to reward zone to cross-session correlation (stability)

def get_pf_locations_by_stability(stab, pfs):
    # Transform place field coordinates to distance from last reward zone
    def get_distance(centers, borders):
        """ Get distance in bin coordinates to next RZ start. """

        def comp_dist(cent, b):
            dist = b - cent
            return np.min(dist[dist > 0])

        if isinstance(centers, float) and np.isnan(centers):
            return np.nan

        try:
            return [comp_dist(c, borders) for c in np.array(centers)]
        except TypeError:
            return comp_dist(centers, borders)

    zones = (hheise_behav.CorridorPattern & 'pattern="training"').rescale_borders(n_bins=80)
    start2start = zones[1, 0] - zones[0, 0]
    end2start = zones[1, 0] - zones[0, 1]

    df_list = []
    for mouse in stab.mouse_id.unique():

        # Skip mice that should be ignored
        if mouse == '121_1':
            continue

        # For each day, get the place field com of each place cell
        for day, coms in pfs[int(mouse.split('_')[0])].items():
            curr_dist = coms.apply(func=get_distance, borders=np.append(zones[:, 1], zones[-1, 1] + start2start))

            if day < 1:
                phase = 'pre'
            elif day > 7:
                phase = 'late'
            else:
                phase = 'early'

            curr_classes = stab[(stab.mouse_id == mouse) & (stab.period == phase)].iloc[0]['classes']
            # Turn stability scores into Series to allow index matching later
            curr_stab = pd.Series(stab[(stab.mouse_id == mouse) & (stab.period == phase)].iloc[0]['stab_vals'], name='stab')
            curr_z_stab = pd.Series(stab[(stab.mouse_id == mouse) & (stab.period == phase)].iloc[0]['stab_z_scores'], name='z_stab')

            # Reformat series of lists into a nice simple numpy array
            stable_dist = curr_dist[curr_classes == 3].explode().dropna().astype(float).to_numpy()
            unstable_dist = curr_dist[curr_classes == 2].explode().dropna().astype(float).to_numpy()

            # Reset index of curr_dist to be incremental (like curr_stab) to enable merging based on index
            curr_dist_exp = curr_dist.reset_index().drop(columns='index').squeeze().explode().rename('dist')
            curr_dist_stab = pd.merge(curr_dist_exp, right=curr_stab, left_index=True, right_index=True)
            curr_dist_stab = pd.merge(curr_dist_stab, right=curr_z_stab, left_index=True, right_index=True)
            curr_dist_stab = curr_dist_stab.dropna(how='all').astype(float)

            # Merge the stability scores to the stable and unstable dist to add them together to the dataframes
            curr_stab_stable = curr_dist_stab.loc[curr_dist_stab.dist.isin(stable_dist)]['stab'].to_numpy()
            curr_z_stab_stable = curr_dist_stab.loc[curr_dist_stab.dist.isin(stable_dist)]['z_stab'].to_numpy()
            curr_stab_unstable = curr_dist_stab.loc[curr_dist_stab.dist.isin(unstable_dist)]['stab'].to_numpy()
            curr_z_stab_unstable = curr_dist_stab.loc[curr_dist_stab.dist.isin(unstable_dist)]['z_stab'].to_numpy()

            # Set distances that are bigger than the distance from the end of one RZ to the start of the next (which means that the distance is within the RZ) to 0
            # This has to happen after matching distance values with isin() in the previous code block
            stable_dist[stable_dist > end2start] = 0
            unstable_dist[unstable_dist > end2start] = 0

            df_list.append(pd.DataFrame(data=dict(mouse_id=int(mouse.split('_')[0]), phase=phase, stab_class='stable',
                                                  dist=stable_dist, stab=curr_stab_stable, z_stab=curr_z_stab_stable)))
            df_list.append(pd.DataFrame(data=dict(mouse_id=int(mouse.split('_')[0]), phase=phase, stab_class='unstable',
                                                  dist=unstable_dist, stab=curr_stab_unstable, z_stab=curr_z_stab_unstable)))

    return pd.concat(df_list, ignore_index=True)

from preprint import stable_unstable_classification as suc
from preprint import data_cleaning as dc

spatial_maps = dc.load_data('spat_dff_maps')
is_pc = dc.load_data('is_pc')
pf_com = pd.read_pickle(r'C:\Users\hheise.UZH\PycharmProjects\Caiman\custom scripts\preprint\Filippo\neural_data\pf_com.pkl')

group = (hheise_grouping.BehaviorGrouping() & 'cluster="fine"' & 'grouping_id=4').get_groups()
group.loc[group.group == "No Deficit", 'group'] = "Sham"

# Get stability classes
stability_classes = suc.classify_stability(is_pc_list=is_pc, spatial_map_list=spatial_maps, for_prism=False,
                                           ignore_lost=True, align_days=True)

df = get_pf_locations_by_stability(stab=stability_classes, pfs=pf_com)
df = df.merge(right=group, on='mouse_id')

plt.figure()
sns.boxplot(data=df, x='phase', y='dist', hue='stab')

sns.catplot(
    data=df, x='group', y='dist', hue='stab',
    col='phase', kind='box')

# Plot mouse averages
mouse_avg = df.groupby(['mouse_id', 'phase', 'stab', 'group']).agg('mean').reset_index()

# Export to Prism for each phase separately (to compare stab distances between groups)
mouse_avg[mouse_avg.phase == 'late'].pivot(index='stab', columns='mouse_id', values='dist').to_clipboard(header=True)

# Compute difference of average distances between stable and unstable cells for each mouse
aggregated = df.groupby(["mouse_id", "phase", "stab"])["dist"].mean().unstack("stab")
aggregated["difference"] = aggregated["stable"] - aggregated["unstable"]
aggregated = aggregated.reset_index()

aggregated.pivot(index='phase', columns='mouse_id', values='difference').loc[['pre', 'early', 'late']].to_clipboard()

# Plot correlation of distance to stability score across groups and phases
def annotate(data, **kws):
    nan_mask = np.isnan(data['stab'])
    r, p = stats.pearsonr(data['dist'][~nan_mask], data['stab'][~nan_mask])
    ax = plt.gca()
    ax.text(.05, .05, 'r={:.2f}, p={:.2g}'.format(r, p),
            transform=ax.transAxes)
#
# g = sns.FacetGrid(df, col="phase",  row="group", col_order=['pre', 'early', 'late'], row_order=['Sham', 'Recovery', 'No Recovery'])
# g.map(sns.scatterplot, 'dist', 'stab')

g = sns.lmplot(data=df, x='dist', y='stab', col="phase",  row="group", height=3, aspect=1,
               col_order=['pre', 'early', 'late'], row_order=['Sham', 'Recovery', 'No Recovery'],
               line_kws={'color': 'green'}, facet_kws={'margin_titles': True})
g.map_dataframe(annotate)


