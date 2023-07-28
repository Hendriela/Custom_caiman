#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 18/07/2023 13:07
@author: hheise

"""
import numpy as np
import pandas as pd
import os

import common_hist
from schema import common_mice, hheise_behav, hheise_hist, hheise_placecell
from util import helper
import functools

os.chdir('.\\custom scripts\\preprint\\Filippo')
#%% Request on 18.07.

# VR Performance over time
no_deficit = [93, 91, 95]
no_deficit_flicker = [111, 114, 116]
recovery = [33, 83, 85, 86, 89, 90, 113]
deficit_no_flicker = [41, 63, 69, 121]
deficit_flicker = [108, 110, 112]
sham_injection = [115, 122]
mice = [*no_deficit, *no_deficit_flicker, *recovery, *deficit_no_flicker, *deficit_flicker, *sham_injection]

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

    pc_ratios = pd.DataFrame((hheise_placecell.PlaceCell & f'mouse_id={mouse}' & 'corridor_type=0' & 'place_cell_id=2').fetch('day', 'place_cell_ratio', as_dict=True))

    # Transform dates into days before surgery
    rel_days = [(d - surgery_day).days for d in days]

    df_wide = pd.DataFrame(dict(mouse_id=mouse, days=days, rel_days=rel_days,
                                blr=perf_lick, si=perf_si, dist=perf_dist))
    df_wide['rel_sess'] = df_wide.index - np.argmax(np.where(df_wide['rel_days'] <= 0, df_wide['rel_days'], -np.inf))

    df_melt = df_wide.melt(id_vars=['mouse_id', 'days', 'rel_days', 'rel_sess'], var_name='metric', value_name='perf')

    met_norm = []
    for metric in df_melt['metric'].unique():
        met = df_melt[df_melt['metric'] == metric]
        met_norm.append(list(met['perf'] / met[(met['rel_sess'] >= -2) & (met['rel_sess'] <= 0)]['perf'].mean()))
    met_norm = [item for sublist in met_norm for item in sublist]
    df_melt['perf_norm'] = met_norm
    dfs.append(df_melt)


df = pd.concat(dfs, ignore_index=True)
df.sort_values(by=['mouse_id', 'metric'], inplace=True, ignore_index=True)
df.to_csv('.\\20230718\\vr_performance.csv', sep=',')

# VR Performance for 2D scatter plot
# take "datas" from behavior_matrix.py
datas[0]['si_binned_run'].to_csv('.\\20230718\\si_scatterplot_norm.csv', sep=',')
datas[1]['si_binned_run'].to_csv('.\\20230718\\si_scatterplot_raw.csv', sep=',')

### Histo data
inj_volume = pd.DataFrame((common_mice.Injection & 'substance_name="microspheres"' & f'mouse_id in {helper.in_query(mice)}').fetch('mouse_id', 'volume', as_dict=True)).rename(columns={'volume': 'inj_volume'})
spheres_extrap = pd.DataFrame((hheise_hist.MicrosphereSummary.Metric & 'metric_name="spheres"' & f'mouse_id in {helper.in_query(mice)}').fetch('mouse_id', 'count_extrap', as_dict=True)).rename(columns={'count_extrap': 'spheres_extrap'})
spheres = pd.DataFrame((hheise_hist.MicrosphereSummary.Metric & 'metric_name="spheres"' & f'mouse_id in {helper.in_query(mice)}').fetch('mouse_id', 'count', as_dict=True)).rename(columns={'count': 'spheres'})
lesion = pd.DataFrame((hheise_hist.MicrosphereSummary.Metric & 'metric_name="auto"' & f'mouse_id in {helper.in_query(mice)}').fetch('mouse_id', 'count', as_dict=True)).rename(columns={'count': 'autofluo'})
lesion_extrap = pd.DataFrame((hheise_hist.MicrosphereSummary.Metric & 'metric_name="auto"' & f'mouse_id in {helper.in_query(mice)}').fetch('mouse_id', 'count_extrap', as_dict=True)).rename(columns={'count_extrap': 'autofluo_extrap'})

df_merged = functools.reduce(lambda left, right: pd.merge(left, right, on=['mouse_id'], how='outer'), [spheres_extrap, spheres, lesion_extrap, lesion, inj_volume])
df_merged.to_csv('.\\20230718\\histology.csv', sep=',')

# PC data (stable/unstable) over time
class_df.to_csv('.\\20230718\\pc_distribution.csv', sep=',')

# PC ratios over time
dfs = []
for mouse in mice:
    # Get day of surgery
    surgery_day = (common_mice.Surgery() & 'surgery_type="Microsphere injection"' & f'mouse_id={mouse}').fetch(
        'surgery_date')[0].date()
    # Get date and performance for each session before the surgery day
    pc_ratios = pd.DataFrame((hheise_placecell.PlaceCell & f'mouse_id={mouse}' & 'corridor_type=0' & 'day<"2022-09-09"'
                              & 'place_cell_id=2').fetch('mouse_id', 'day', 'place_cell_ratio', as_dict=True))
    n_pcs = pd.DataFrame((hheise_placecell.PlaceCell.ROI & f'mouse_id={mouse}' & 'corridor_type=0' & 'day<"2022-09-09"' & 'place_cell_id=2' & 'is_place_cell').fetch('day'))[0].value_counts()
    n_pcs = pd.DataFrame({'day': n_pcs.index, 'n_pcs': n_pcs.values})
    pc_ratios['rel_days'] = [(d - surgery_day).days for d in pc_ratios['day']]
    pc_ratios['place_cell_ratio'] *= 100
    pc_ratio1 = pd.merge(pc_ratios, n_pcs, how='outer', on='day')
    dfs.append(pc_ratio1)
pc_ratios = pd.concat(dfs, ignore_index=True)
pc_ratios.sort_values(by=['mouse_id', 'rel_days'], inplace=True, ignore_index=True)
pc_ratios.fillna(0, inplace=True)
pc_ratios.to_csv('.\\20230718\\pc_ratios.csv', sep=',')

#%% Request on 26.07.2023

# Binary licks for each mouse/session

dfs = []
for mouse in mice:

    # Get day of surgery
    surgery_day = (common_mice.Surgery() & 'surgery_type="Microsphere injection"' & f'mouse_id={mouse}').fetch(
        'surgery_date')[0].date()

    # Get binary lick data
    pks = (hheise_behav.VRPerformance & 'username="hheise"' & f'mouse_id={mouse}').fetch('KEY')
    binary_licks = []
    for key in pks:
        trials = (hheise_behav.VRSession & key).get_normal_trials()
        # Bin licking data from these trials into one array and binarize per-trial
        data = [(hheise_behav.VRSession.VRTrial & key & f'trial_id={idx}').get_binned_licking(bin_size=1) for idx in
                trials]
        data = np.vstack(data)
        data[data > 0] = 1  # we only care about that a bin had a lick, not how many
        binary_licks.append(data)

    # Transform dates into days before surgery
    rel_days = np.array([(pk['day'] - surgery_day).days for pk in pks])

    if 3 not in rel_days:
        rel_days[(rel_days == 2) | (rel_days == 4)] = 3
    rel_days[(rel_days == 5) | (rel_days == 6) | (rel_days == 7)] = 6
    rel_days[(rel_days == 8) | (rel_days == 9) | (rel_days == 10)] = 9
    rel_days[(rel_days == 11) | (rel_days == 12) | (rel_days == 13)] = 12
    rel_days[(rel_days == 14) | (rel_days == 15) | (rel_days == 16)] = 15
    rel_days[(rel_days == 17) | (rel_days == 18) | (rel_days == 19)] = 18
    rel_days[(rel_days == 20) | (rel_days == 21) | (rel_days == 22)] = 21
    rel_days[(rel_days == 23) | (rel_days == 24) | (rel_days == 25)] = 24
    rel_days[(rel_days == 26) | (rel_days == 27) | (rel_days == 28)] = 27

    rel_sess = np.arange(len(rel_days)) - np.argmax(np.where(rel_days <= 0, rel_days, -np.inf))

    rel_days[(-5 < rel_sess) & (rel_sess < 1)] = np.arange(-4, 1)

    df_wide = pd.DataFrame(dict(mouse_id=mouse, days=[pk['day'] for pk in pks], rel_days=rel_days, rel_sess=rel_sess, licks=binary_licks))

    dfs.append(df_wide)

df = pd.concat(dfs, ignore_index=True)

df = df[df['rel_sess'] >= -4]
df = df[df['rel_days'] != 1]
df = df[df['rel_days'] != 2]
df = df[df['rel_days'] != 4]
df = df[df['rel_sess'] <= 9]

df_filt = df[['mouse_id', 'rel_days', 'licks']]
df_pivot = df_filt.pivot(columns='rel_days', index='mouse_id', values='licks')

df_pivot.to_pickle('.\\20230726\\bin_licks.pickle')

# Sphere location

# Non-summed locations
mic = pd.DataFrame(hheise_hist.Microsphere().fetch('mouse_id', 'hemisphere', 'structure_id', 'lesion', 'spheres', as_dict=True))
sum_spheres_df = mic.groupby(['mouse_id', 'hemisphere', 'structure_id', 'lesion']).agg({'spheres': 'sum'}).reset_index()
sum_spheres_df.loc[sum_spheres_df['mouse_id'] >= 108, 'lesion'] = np.nan
sum_spheres_df['acronym'] = [(common_hist.Ontology & f'structure_id={id_}').fetch1('acronym') for id_ in sum_spheres_df['structure_id']]
sum_spheres_df['full_name'] = [(common_hist.Ontology & f'structure_id={id_}').fetch1('full_name') for id_ in sum_spheres_df['structure_id']]
sum_spheres_df = sum_spheres_df[sum_spheres_df['mouse_id'] != 94]
sum_spheres_df.to_csv('.\\20230726\\raw_locations.csv', sep=',', index=False)

# Summed locations
columns = ['spheres', 'spheres_rel', 'spheres_lesion', 'spheres_lesion_rel', 'lesion_vol', 'lesion_vol_rel',
           'spheres_extrap', 'spheres_rel_extrap', 'spheres_lesion_extrap', 'spheres_lesion_rel_extrap',
           'region', 'mouse_id']

# Old grouping
df = hheise_hist.Microsphere().get_structure_groups(grouping={'cognitive': ['HPF', 'PL', 'ACA', 'ILA', 'RSP', 'PERI'],
                                                              'neocortex': ['FRP', 'MO', 'OLF', 'SS', 'GU', 'VISC',
                                                                            'AUD', 'VIS', 'ORB', 'AI', 'PTLp', 'TEa',
                                                                            'ECT'],
                                                              'thalamus': ['TH'],
                                                              'basal_ganglia': ['CP', 'ACB', 'FS', 'LSX', 'sAMY', 'PAL'],
                                                              'brainstem': ['MB', 'HB']},
                                                    columns=columns)
df = df[df['mouse_id'] != 94]
df.to_csv('.\\20230726\\old_grouping.csv', sep=',', index=False)

# Hippocampal formation
df = hheise_hist.Microsphere().get_structure_groups(grouping={'hippocampus': ['HPF'],
                                                              'cortex': ['Isocortex', 'OLF'],
                                                              'thalamus': ['TH'],
                                                              'cerebral_nuclei': ['CNU'],
                                                              'brainstem': ['MB', 'HB'],
                                                              'white_matter': ['fiber tracts']},
                                                    columns=columns)
df = df[df['mouse_id'] != 94]
df.to_csv('.\\20230726\\hippocampus.csv', sep=',', index=False)

# Only Hippocampal formation
df = hheise_hist.Microsphere().get_structure_groups(grouping={'hippocampal_formation': ['HPF']},
                                                    columns=columns)
df = df[df['mouse_id'] != 94]
df.to_csv('.\\20230726\\hippocampal_formation.csv', sep=',', index=False)

# Hippocampal subregions
df = hheise_hist.Microsphere().get_structure_groups(grouping={'subiculum': ['SUB', 'PRE', 'PAR', 'POST'],
                                                              'entorhinal': ['ENT'],
                                                              'dentate_gyrus': ['DG'],
                                                              'ca1': ['CA1'],
                                                              'ca2': ['CA2'],
                                                              'ca3': ['CA3'],
                                                              },
                                                    columns=columns)
df = df[df['mouse_id'] != 94]
df.to_csv('.\\20230726\\hippocampal_subregions.csv', sep=',', index=False)

# Silasi grouping
df = hheise_hist.Microsphere().get_structure_groups(grouping={'hippocampus': ['HPF'],
                                                              'white_matter': ['fiber tracts'],
                                                              'striatum': ['STR'],
                                                              'thalamus': ['TH'],
                                                              'neocortex': ['Isocortex'],
                                                              },
                                                    columns=columns)
df = df[df['mouse_id'] != 94]
df.to_csv('.\\20230726\\literature.csv', sep=',', index=False)

# Functional grouping (?)
df = hheise_hist.Microsphere().get_structure_groups(grouping={'cognitive': ['HPF', 'PL', 'ACA', 'ILA', 'RSP', 'PERI',
                                                                            ],
                                                              'cortex': ['Isocortex', 'OLF'],
                                                              'thalamus': ['TH'],
                                                              'basal_ganglia': ['CNU'],
                                                              'brainstem': ['MB', 'HB'],
                                                              'white_matter': ['fiber tracts']},
                                                    columns=columns)
df = df[df['mouse_id'] != 94]
df.to_csv('.\\20230726\\hippocampus.csv', sep=',', index=False)
