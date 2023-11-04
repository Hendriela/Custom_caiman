#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 03/11/2023 09:26
@author: hheise

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from preprint import data_cleaning as dc
from preprint import placecell_heatmap_transition_functions as func


def quantify_place_cell_transitions(pf_list, pc_list, align_days=False, day_diff=3, shuffle=None):

    pc_transitions = []

    for mouse_idx, mouse in enumerate(pc_list):

        mouse_id = list(mouse.keys())[0]
        # if mouse_id == '41_1':
        #     break
        # Extract data and align days if necessary
        rel_days = np.array(mouse[mouse_id][1])

        if align_days:
            if 3 not in rel_days:
                rel_days[(rel_days == 2) | (rel_days == 4)] = 3
            rel_days[(rel_days == 5) | (rel_days == 6) | (rel_days == 7)] = 6
            rel_days[(rel_days == 8) | (rel_days == 9) | (rel_days == 10)] = 9
            rel_days[(rel_days == 11) | (rel_days == 12) | (rel_days == 13)] = 12
            rel_days[(rel_days == 14) | (rel_days == 15) | (rel_days == 16)] = 15
            rel_days[(rel_days == 17) | (rel_days == 18) | (rel_days == 19)] = 18
            rel_days[(rel_days == 20) | (rel_days == 21) | (rel_days == 22)] = 21
            rel_days[(rel_days == 23) | (rel_days == 24) | (rel_days == 25)] = 24
            if 28 not in rel_days:
                rel_days[(rel_days == 26) | (rel_days == 27) | (rel_days == 28)] = 27
            rel_sess = np.arange(len(rel_days)) - np.argmax(np.where(rel_days <= 0, rel_days, -np.inf))
            rel_days[(-5 < rel_sess) & (rel_sess < 1)] = np.arange(-np.sum((-5 < rel_sess) & (rel_sess < 1)) + 1, 1)

        pc_data = mouse[mouse_id][0]
        pc_data = pd.DataFrame(pc_data, columns=rel_days)
        pf_data = pf_list[mouse_idx][mouse_id][0]
        pf_data.columns = rel_days
        pf_data = pf_data.reset_index().drop(columns=['index'])     # Use continuous index that matches pc_data

        # Ignore day 1
        rel_days = rel_days[rel_days != 1]
        pf_data = pf_data.loc[:, pf_data.columns != 1]
        pc_data = pc_data.loc[:, pc_data.columns != 1]

        if shuffle is not None:
            iterations = shuffle
        else:
            iterations = 1

        pc_trans = {'pre': np.zeros((iterations, 3, 3)), 'early': np.zeros((iterations, 3, 3)), 'late': np.zeros((iterations, 3, 3))}
        stable_pc_trans = {'pre': np.zeros((iterations, 4, 4)), 'early': np.zeros((iterations, 4, 4)), 'late': np.zeros((iterations, 4, 4))}

        for i in range(iterations):

            rng = np.random.default_rng()   # Initialize the random generator

            # Loop through days and get place cell transitions between sessions that are 3 days apart
            for day_idx, day in enumerate(rel_days):

                next_day_idx = np.where(rel_days == day + day_diff)[0]

                # If a session 3 days later exists, get place cell transitions
                if len(next_day_idx) == 1:

                    # General PC - non-PC transitions
                    # Add 1 to the pc data to include "Lost" cells
                    day1_pc = pc_data.iloc[:, day_idx].to_numpy() + 1
                    day1_pc = np.nan_to_num(day1_pc).astype(int)
                    day2_pc = pc_data.iloc[:, next_day_idx].to_numpy().squeeze() + 1
                    day2_pc = np.nan_to_num(day2_pc).astype(int)

                    if shuffle is not None:

                        # mat_true = func.transition_matrix(mask1=day1_pc, mask2=day2_pc, num_classes=3, percent=False)
                        # mat_shuff = []
                        # for i in range(shuffle):
                        #     day2_pc_shuff = np.random.default_rng().permutation(day2_pc)
                        #     mat_shuff.append(func.transition_matrix(mask1=day1_pc, mask2=day2_pc_shuff, num_classes=3, percent=False))
                        # mat = np.stack(mat_shuff)

                        mat = func.transition_matrix(mask1=day1_pc, mask2=rng.permutation(day2_pc),
                                                     num_classes=3, percent=False)

                        # mat_shuff_mean = np.mean(mat_shuff, axis=0)
                        #
                        # # Plot shuffled distribution and true value
                        # nrows = mat_true.shape[0]
                        # ncols = mat_true.shape[1]
                        # fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
                        # for i in range(nrows):
                        #     for j in range(ncols):
                        #         # sns.violinplot(y=mat_shuff[:, i, j], ax=ax[i, j], cut=0)
                        #         sns.stripplot(y=mat_shuff[:, i, j], ax=ax[i, j], alpha=0.5)
                        #         ax[i, j].axhline(mat_true[i, j], color='red')
                        #         perc = (np.sum(mat_shuff[:, i, j] < mat_true[i, j]) / len(mat_shuff[:, i, j])) * 100
                        #         ax[i, j].text(0.05, 0.95, perc, transform=ax[i, j].transAxes, verticalalignment='top',
                        #                       horizontalalignment='left')
                    else:
                        mat = func.transition_matrix(mask1=day1_pc, mask2=day2_pc, num_classes=3, percent=False)

                    if rel_days[next_day_idx] <= 0:
                        pc_trans['pre'][i] = pc_trans['pre'][i] + mat
                    elif rel_days[next_day_idx] <= 7:
                        pc_trans['early'][i] = pc_trans['early'][i] + mat
                    else:
                        pc_trans['late'][i] = pc_trans['late'][i] + mat

                    # Split PC-PC into stable (pf_idx overlap) and unstable PC transitions
                    pc_pc_idx = (day1_pc + day2_pc) == 4
                    pf1 = pf_data.loc[pc_pc_idx].iloc[:, day_idx]
                    pf2 = pf_data.loc[pc_pc_idx].iloc[:, next_day_idx[0]]

                    for row_idx in pf1.index:
                        pf_1 = pf1.loc[row_idx]
                        pf_2 = pf2.loc[row_idx]

                        overlapping_pf = False
                        for place_field_1 in pf_1:
                            for place_field_2 in pf_2:
                                if np.any([i in place_field_1 for i in place_field_2]):
                                    overlapping_pf = True

                        if overlapping_pf:
                            # Class 3 means that the place cell retains at least one place field
                            day1_pc[row_idx] = 3
                            day2_pc[row_idx] = 3

                    mat = func.transition_matrix(mask1=day1_pc, mask2=day2_pc, num_classes=4, percent=False)
                    if rel_days[next_day_idx] <= 0:
                        stable_pc_trans['pre'] = stable_pc_trans['pre'] + mat
                    elif rel_days[next_day_idx] <= 7:
                        stable_pc_trans['early'] = stable_pc_trans['early'] + mat
                    else:
                        stable_pc_trans['late'] = stable_pc_trans['late'] + mat

        pc_transitions.append(pd.DataFrame([dict(mouse_id=int(mouse_id.split('_')[0]),
                                                 pc_pre=pc_trans['pre'].squeeze(), pc_early=pc_trans['early'].squeeze(), pc_late=pc_trans['late'].squeeze(),
                                                 stab_pc_pre=stable_pc_trans['pre'].squeeze(), stab_pc_early=stable_pc_trans['early'].squeeze(), stab_pc_late=stable_pc_trans['late'].squeeze())]))

    return pd.concat(pc_transitions, ignore_index=True)


def transition_matrix_to_prism(matrix_df: pd.DataFrame, phase, include_lost=False, with_stable=False, norm='rows'):

    if with_stable:
        col = f'stab_pc_{phase}'
    else:
        col = f'pc_{phase}'

    dicts = []
    for _, row in matrix_df.iterrows():
        if include_lost:
            if norm in ['rows', 'forward']:
                mat = row[col] / np.sum(row[col], axis=1)[..., np.newaxis] * 100
            elif norm in ['cols', 'backward']:
                mat = row[col] / np.sum(row[col], axis=0) * 100
            elif norm in ['all']:
                mat = row[col] / np.sum(row[col]) * 100
            else:
                raise NotImplementedError

            ### Include "lost"
            if with_stable:
                dicts.append(dict(trans='lost > lost', mouse_id=row['mouse_id'], perc=mat[0, 0]))
                dicts.append(dict(trans='lost > non-coding', mouse_id=row['mouse_id'], perc=mat[0, 1]))
                dicts.append(dict(trans='lost > remapped', mouse_id=row['mouse_id'], perc=mat[0, 2]))
                dicts.append(dict(trans='lost > stable', mouse_id=row['mouse_id'], perc=mat[0, 3]))
                dicts.append(dict(trans='non-coding > lost', mouse_id=row['mouse_id'], perc=mat[1, 0]))
                dicts.append(dict(trans='non-coding > non-coding', mouse_id=row['mouse_id'], perc=mat[1, 1]))
                dicts.append(dict(trans='non-coding > remapped', mouse_id=row['mouse_id'], perc=mat[1, 2]))
                dicts.append(dict(trans='non-coding > place_cell', mouse_id=row['mouse_id'], perc=mat[1, 3]))
                dicts.append(dict(trans='remapped > lost', mouse_id=row['mouse_id'], perc=mat[2, 0]))
                dicts.append(dict(trans='remapped > non-coding', mouse_id=row['mouse_id'], perc=mat[2, 1]))
                dicts.append(dict(trans='remapped > remapped', mouse_id=row['mouse_id'], perc=mat[2, 2]))
                dicts.append(dict(trans='remapped > stable', mouse_id=row['mouse_id'], perc=mat[2, 3]))
                dicts.append(dict(trans='stable > lost', mouse_id=row['mouse_id'], perc=mat[3, 0]))
                dicts.append(dict(trans='stable > non-coding', mouse_id=row['mouse_id'], perc=mat[3, 1]))
                dicts.append(dict(trans='stable > unstable', mouse_id=row['mouse_id'], perc=mat[3, 2]))
                dicts.append(dict(trans='stable > stable', mouse_id=row['mouse_id'], perc=mat[3, 3]))
            else:
                dicts.append(dict(trans='lost > lost', mouse_id=row['mouse_id'], perc=mat[0, 0]))
                dicts.append(dict(trans='lost > non-coding', mouse_id=row['mouse_id'], perc=mat[0, 1]))
                dicts.append(dict(trans='lost > place_cell', mouse_id=row['mouse_id'], perc=mat[0, 2]))
                dicts.append(dict(trans='non-coding > lost', mouse_id=row['mouse_id'], perc=mat[1, 0]))
                dicts.append(dict(trans='non-coding > non-coding', mouse_id=row['mouse_id'], perc=mat[1, 1]))
                dicts.append(dict(trans='non-coding > place_cell', mouse_id=row['mouse_id'], perc=mat[1, 2]))
                dicts.append(dict(trans='place_cell > lost', mouse_id=row['mouse_id'], perc=mat[2, 0]))
                dicts.append(dict(trans='place_cell > non-coding', mouse_id=row['mouse_id'], perc=mat[2, 1]))
                dicts.append(dict(trans='place_cell > place_cell', mouse_id=row['mouse_id'], perc=mat[2, 2]))

        else:
            if norm in ['rows', 'forward']:
                mat = row[col][1:, 1:] / np.sum(row[col][1:, 1:], axis=1)[..., np.newaxis] * 100
            elif norm in ['cols', 'backward']:
                mat = row[col][1:, 1:] / np.sum(row[col][1:, 1:], axis=0) * 100
            elif norm in ['all']:
                mat = row[col][1:, 1:] / np.sum(row[col][1:, 1:]) * 100
            else:
                raise NotImplementedError

            ### Exclude "lost"
            if with_stable:
                dicts.append(dict(trans='non-coding > non-coding', mouse_id=row['mouse_id'], perc=mat[0, 0]))
                dicts.append(dict(trans='non-coding > remapped', mouse_id=row['mouse_id'], perc=mat[0, 1]))
                dicts.append(dict(trans='non-coding > stable', mouse_id=row['mouse_id'], perc=mat[0, 2]))
                dicts.append(dict(trans='remapped > non-coding', mouse_id=row['mouse_id'], perc=mat[1, 0]))
                dicts.append(dict(trans='remapped > remapped', mouse_id=row['mouse_id'], perc=mat[1, 1]))
                dicts.append(dict(trans='remapped > stable', mouse_id=row['mouse_id'], perc=mat[1, 2]))
                dicts.append(dict(trans='stable > non-coding', mouse_id=row['mouse_id'], perc=mat[2, 0]))
                dicts.append(dict(trans='stable > remapped', mouse_id=row['mouse_id'], perc=mat[2, 1]))
                dicts.append(dict(trans='stable > stable', mouse_id=row['mouse_id'], perc=mat[2, 2]))
            else:
                dicts.append(dict(trans='non-coding > non-coding', mouse_id=row['mouse_id'], perc=mat[0, 0]))
                dicts.append(dict(trans='non-coding > place_cell', mouse_id=row['mouse_id'], perc=mat[0, 1]))
                dicts.append(dict(trans='place_cell > non-coding', mouse_id=row['mouse_id'], perc=mat[1, 0]))
                dicts.append(dict(trans='place_cell > place_cell', mouse_id=row['mouse_id'], perc=mat[1, 1]))

    df = pd.DataFrame(dicts).pivot(index='trans', columns='mouse_id', values='perc')
    if with_stable:
        if norm in ['rows', 'forward']:
            df = df.reindex(['non-coding > non-coding', 'non-coding > remapped', 'non-coding > stable',
                             'remapped > non-coding', 'remapped > remapped', 'remapped > place_cell',
                             'place_cell > non-coding', 'place_cell > remapped', 'place_cell > place_cell'])
        elif norm in ['cols', 'backward']:
            df = df.reindex(['non-coding > non-coding', 'remapped > non-coding', 'stable > non-coding',
                             'non-coding > remapped', 'remapped > remapped', 'stable > remapped',
                             'non-coding > stable', 'remapped > stable', 'stable > stable'])
        elif norm in ['all']:
            df = df.reindex(['non-coding > non-coding', 'non-coding > remapped', 'non-coding > stable',
                             'remapped > non-coding', 'remapped > remapped', 'remapped > place_cell',
                             'place_cell > non-coding', 'place_cell > remapped', 'place_cell > place_cell'])
    else:
        if norm in ['rows', 'forward']:
            df = df.reindex(['non-coding > non-coding', 'non-coding > place_cell',
                             'place_cell > non-coding', 'place_cell > place_cell'])
        elif norm in ['cols', 'backward']:
            df = df.reindex(['non-coding > non-coding', 'place_cell > non-coding',
                             'non-coding > place_cell', 'place_cell > place_cell'])
        elif norm in ['all']:
            df = df.reindex(['non-coding > non-coding', 'non-coding > place_cell',
                             'place_cell > non-coding', 'place_cell > place_cell'])

    return df


#%% Function calls
pf_idx = dc.load_data('pf_idx')
is_pc = dc.load_data('is_pc')

# pc_transition = quantify_place_cell_transitions(pf_list=pf_idx, pc_list=is_pc)
pc_transition = quantify_place_cell_transitions(pf_list=pf_idx, pc_list=is_pc)
pc_transition_rng = quantify_place_cell_transitions(pf_list=pf_idx, pc_list=is_pc, shuffle=20)

transition_matrix_to_prism(matrix_df=pc_transition, phase='late', include_lost=False, with_stable=False,
                           norm='backward').to_clipboard(index=True, header=False)
