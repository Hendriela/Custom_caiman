#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 18/05/2022 15:18
@author: hheise

Analysis script for microsphere histology data with DataJoint
"""
import matplotlib
matplotlib.use('TkAgg')

from schema import hheise_hist, common_hist, common_mice, hheise_behav
from util import helper
import numpy as np
import pandas as pd
import seaborn as sn

from matplotlib import pyplot as plt


# Group regions (acronyms of largest substructures that should be grouped during analysis)

### BRAIN REGIONS ###
cognitive = ['HPF', 'PL', 'ACA', 'ILA', 'RSP', 'PERI']
neocortex = ['FRP', 'MO', 'OLF', 'SS', 'GU', 'VISC', 'AUD', 'VIS', 'ORB', 'AI', 'PTLp', 'TEa', 'ECT']
thalamus = ['TH']
basal_ganglia = ['CP', 'ACB', 'FS', 'LSX', 'sAMY', 'PAL']
stem = ['MB', 'HB']
group = [cognitive, neocortex, thalamus, basal_ganglia, stem]
group_names = ['cognitive', 'neocortex', 'thalamus', 'basal_ganglia', 'brainstem', 'other']

### CORTICAL - SUBCORTICAL ###
# cortical = ['Isocortex', 'OLF']
# group = [cortical]
# group_names = ['cortical', 'subcortical']

# Check if any structure is included in more than one group (the following function should print the duplicates)
# And select structures of 'others'
others = common_hist.Ontology().validate_grouping(group)

# To test the function, append "others" to the group and run the function again. The returned list should be empty.
group.append(others)
assert len(common_hist.Ontology().validate_grouping(group)) == 0

# Get Microsphere data from these combined structures
total = []
for curr_group, group_name in zip(group, group_names):
    # Get microsphere data from the current structure grouping. Ignore Nones if no spheres are in the structure.
    single_regions = [hheise_hist.Microsphere().get_structure(struc) for struc in curr_group if
                      hheise_hist.Microsphere().get_structure(struc) is not None]

    # Add values of individual regions together to get a common DataFrame of all structures in the grouping
    group_data = single_regions[0]
    for single_reg in single_regions[1:]:
        group_data = group_data.add(single_reg, fill_value=0)[group_data.columns.to_list()]

    # Convert total lesion size to value relative to structure group volume
    rel_img_vol = (common_hist.Histology & f'mouse_id in {helper.in_query(group_data.index)}').fetch('rel_imaged_vol')
    tot_struct_vol = (common_hist.Ontology & f'acronym in {helper.in_query(cognitive)}').fetch('volume').sum()
    group_data['lesion_vol_rel'] = group_data['lesion_vol'] / tot_struct_vol * 100  # make it percentage for nicer values
    group_data['lesion_vol_rel'] /= rel_img_vol     # Extrapolate via relative imaged volume to whole structure

    # Add column of group name to the data points, and make the mouse ID a separate column to make later merge easier
    group_data['region'] = group_name
    group_data['mouse_id'] = group_data.index
    total.append(group_data)

# Concatenate groups together into one dataframe
df = pd.concat(total)
df = df.rename(columns={'spheres_total': 'spheres'})

# Plot sphere and lesion distribution
plotting_order = ['spheres', 'spheres_lesion', 'auto', 'auto_spheres', 'gfap', 'gfap_spheres']

for i, col in enumerate(plotting_order):
    fig, ax = plt.subplots(1, 2, figsize=(15, 8), sharex='all')
    sn.boxplot(x='region', y=col, data=df, fliersize=0, ax=ax[0])
    g = sn.stripplot(x='region', y=col, hue='mouse_id', data=df, ax=ax[0])
    g.legend_.remove()
    sn.boxplot(x='region', y=col+'_rel', data=df, fliersize=0, ax=ax[1])
    g = sn.stripplot(x='region', y=col+'_rel', hue='mouse_id', data=df, ax=ax[1])
    sn.move_legend(ax[1], "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f'W:\\Neurophysiology-Storage1\\Wahl\\Hendrik\\PhD\\Data\\analysis\\histology\\by_regions\\{col}.png')
    plt.close()


### Screen for regions where damage/spheres are best correlated with performance drop

# Get single value for performance drop: Ratio between week before and week after lesion

# # Only work with mice that have histology data on record
# mouse_ids = df['mouse_id'].unique()
#
# for mouse in mouse_ids:
#     # get date of microsphere injection
#     surgery_day = (common_mice.Surgery() & 'surgery_type="Microsphere injection"' & f'mouse_id={mouse}').fetch1('surgery_date')
#
#     # Get the 5 VR sessions before this day
#     pre_dates = (hheise_behav.VRPerformance & f'mouse_id={mouse}' & f'day < "{surgery_day.date()}"').fetch('day')[-5:]
#     self = hheise_behav.VRPerformance & f'day in {helper.in_query(pre_dates)}'

# Add performance grouping to histology data
no_deficit = [91, 94, 95]
recovery = [33, 38, 83, 85, 86, 89, 90]
no_recovery = [41, 63, 69, 93]
mice = [*no_deficit, *recovery, *no_recovery]
grouping = pd.DataFrame(data=dict(mouse_id=mice, group=[*['no_deficit']*len(no_deficit), *['recovery']*len(recovery),
                                                        *['no_recovery']*len(no_recovery)]))
df_grouped = df.merge(grouping)

plotting_order = ['spheres', 'spheres_lesion', 'auto', 'auto_spheres', 'gfap', 'gfap_spheres', 'lesion_vol']

for i, col in enumerate(plotting_order):
    fig, ax = plt.subplots(1, 2, figsize=(15, 8), sharex='all')
    sn.boxplot(x='region', y=col, hue='group', data=df_grouped, ax=ax[0],
               hue_order=['no_deficit', 'recovery', 'no_recovery'], palette=["green", "blue", "red"])
    # g = sn.stripplot(x='region', y=col, hue='group', data=df_grouped, ax=ax[0])
    # g.legend_.remove()
    sn.boxplot(x='region', y=col+'_rel', hue='group', data=df_grouped, ax=ax[1],
               hue_order=['no_deficit', 'recovery', 'no_recovery'], palette=["green", "blue", "red"])
    # g = sn.stripplot(x='region', y=col+'_rel', hue='group', data=df_grouped, ax=ax[1])
    # sn.move_legend(ax[1], "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f'W:\\Neurophysiology-Storage1\\Wahl\\Hendrik\\PhD\\Data\\analysis\\histology\\by_performance\\{col}.png')
    plt.close()

# Save data for spheres and lesion_vol_rel in txt file for Prism export
spheres = np.zeros((len(group_names), 7*3)) * np.nan
lesion_vol_rel = np.zeros((len(group_names), 7*3)) * np.nan
auto_rel = np.zeros((len(group_names), 7*3)) * np.nan
header = ['']*7*3

for group_idx, curr_group in enumerate([no_deficit, recovery, no_recovery]):
    for idx, mouse in enumerate(curr_group):
        for region_idx, region in enumerate(group_names):
            try:
                spheres[region_idx, idx+(group_idx*7)] = df_grouped[(df_grouped['mouse_id'] == mouse) &
                                                                      (df_grouped['region'] == region)]['spheres']
                lesion_vol_rel[region_idx, idx+(group_idx*7)] = df_grouped[(df_grouped['mouse_id'] == mouse) &
                                                                      (df_grouped['region'] == region)]['lesion_vol_rel']
                auto_rel[region_idx, idx+(group_idx*7)] = df_grouped[(df_grouped['mouse_id'] == mouse) &
                                                                      (df_grouped['region'] == region)]['auto_rel']
            except ValueError: # This catches an error if no data from that region is available for the mouse
                pass
            header[idx+(group_idx*7)] = str(mouse)

np.savetxt(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Posters\FENS 2022\histo_group_regions_spheres.csv', spheres,
           delimiter='\t', header='\t'.join(header))

np.savetxt(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Posters\FENS 2022\histo_group_regions_lesion.csv', lesion_vol_rel,
           delimiter='\t', header='\t'.join(header))

np.savetxt(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Posters\FENS 2022\histo_group_regions_autofluo_rel.csv', auto_rel,
           delimiter='\t', header='\t'.join(header))
