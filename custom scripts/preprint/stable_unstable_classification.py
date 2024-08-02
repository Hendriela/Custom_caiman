#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 22/08/2023 14:08
@author: hheise

"""
import os
import numpy as np
import pandas as pd
from typing import Optional, Iterable, List, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns

from schema import hheise_grouping, hheise_placecell, common_match, hheise_behav
from preprint import data_cleaning as dc
from preprint import placecell_heatmap_transition_functions as func


def classify_stability(is_pc_list, spatial_map_list, for_prism=True, ignore_lost=False, align_days=False, aligned_column_names=False):

    mice = [next(iter(dic.keys())) for dic in spatial_map_list]

    dfs = []
    for i, mouse in enumerate(mice):
        rel_dates = np.array(is_pc_list[i][mouse][1])

        if align_days:
            if 3 not in rel_dates:
                rel_dates[(rel_dates == 2) | (rel_dates == 4)] = 3
            rel_dates[(rel_dates == 5) | (rel_dates == 6) | (rel_dates == 7)] = 6
            rel_dates[(rel_dates == 8) | (rel_dates == 9) | (rel_dates == 10)] = 9
            rel_dates[(rel_dates == 11) | (rel_dates == 12) | (rel_dates == 13)] = 12
            rel_dates[(rel_dates == 14) | (rel_dates == 15) | (rel_dates == 16)] = 15
            rel_dates[(rel_dates == 17) | (rel_dates == 18) | (rel_dates == 19)] = 18
            rel_dates[(rel_dates == 20) | (rel_dates == 21) | (rel_dates == 22)] = 21
            rel_dates[(rel_dates == 23) | (rel_dates == 24) | (rel_dates == 25)] = 24
            if 28 not in rel_dates:
                rel_dates[(rel_dates == 26) | (rel_dates == 27) | (rel_dates == 28)] = 27

            # Uncomment this code if prestroke days should also be aligned (probably not good nor necessary)
            rel_sess = np.arange(len(rel_dates)) - np.argmax(np.where(rel_dates <= 0, rel_dates, -np.inf))
            rel_dates[(-5 < rel_sess) & (rel_sess < 1)] = np.arange(-np.sum((-5 < rel_sess) & (rel_sess < 1)) + 1, 1)

        mask_pre = rel_dates <= 0
        mask_early = (0 < rel_dates) & (rel_dates <= 7)
        mask_late = rel_dates > 7

        classes_pre, stability_thresh = func.quantify_stability_split(is_pc_arr=is_pc_list[i][mouse][0][:, mask_pre],
                                                                      spat_dff_arr=spatial_map_list[i][mouse][0][:, mask_pre],
                                                                      rel_days=rel_dates[mask_pre], mouse_id=mouse)
        classes_early = func.quantify_stability_split(is_pc_arr=is_pc_list[i][mouse][0][:, mask_early],
                                                      spat_dff_arr=spatial_map_list[i][mouse][0][:, mask_early],
                                                      rel_days=rel_dates[mask_early], stab_thresh=stability_thresh)
        classes_late = func.quantify_stability_split(is_pc_arr=is_pc_list[i][mouse][0][:, mask_late],
                                                     spat_dff_arr=spatial_map_list[i][mouse][0][:, mask_late],
                                                     rel_days=rel_dates[mask_late], stab_thresh=stability_thresh)

        df = pd.concat([func.summarize_quantification(classes_pre, 'pre'),
                        func.summarize_quantification(classes_early, 'early'),
                        func.summarize_quantification(classes_late, 'late')])

        df['mouse_id'] = mouse
        df['classes'] = [classes_pre, classes_early, classes_late]
        dfs.append(df)

    class_df = pd.concat(dfs, ignore_index=True)

    if ignore_lost:

        def remove_lost_cells(row):
            n_cells = row[['n1', 'n2', 'n3']].sum()
            row['n0_r'] = np.nan
            row['n1_r'] = (row['n1'] / n_cells) * 100
            row['n2_r'] = (row['n2'] / n_cells) * 100
            row['n3_r'] = (row['n3'] / n_cells) * 100

            return row

        class_df = class_df.apply(remove_lost_cells, axis=1)

    if for_prism:
        return func.pivot_classdf_prism(class_df, percent=True, col_order=['pre', 'early', 'late'])
    else:
        return class_df


def stability_sankey(df: pd.DataFrame, shuffle: Union[bool, int] = False, return_shuffle_avg=False, grouping_id: int = 4, return_group_avg=False,
                     directory: Optional[str] = r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Papers\preprint\class_quantification\sankey_plot'):
    """
    Export a "stability_classes" Dataframe for Plotly`s Sankey diagram.

    Args:
        df:                 Stability_classes Dataframe, output from classify_stability().
        shuffle:            Whether to shuffle the cell identities to create shuffled chance level data.
        return_shuffle_avg: Whether to average matrices across shuffles before returning.
        grouping_id:        Parameter ID of the mouse behavioral grouping to be used.
        return_group_avg:   If True, matrices of single mice are averaged across groups from grouping_id.
        directory:          Directory where to save the CSV files for the Sankey plot. If None, dataframe is returned and not saved to file.
    """

    # Load behavioral grouping from Datajoint
    coarse = (hheise_grouping.BehaviorGrouping & f'grouping_id={grouping_id}' & 'cluster="coarse"').get_groups(as_frame=False)
    fine = (hheise_grouping.BehaviorGrouping & f'grouping_id={grouping_id}' & 'cluster="fine"').get_groups(as_frame=False)

    # Combine dicts, adapt names to file system and remove 121
    groups = {'control': coarse['Control'], 'stroke_with121': coarse['Stroke'],
              'stroke': [x for x in coarse['Stroke'] if x != 121],
              'sham': fine['Sham'], 'no_deficit': fine['No Deficit'], 'recovery': fine['Recovery'],
              'no_recovery_with121': fine['No Recovery'], 'no_recovery': [x for x in fine['No Recovery'] if x != 121]}

    print(f'Using following groups:\n{groups}')

    # Transform cell identity arrays into transition matrices
    if (shuffle is not None) and shuffle:
        iterations = shuffle
    else:
        iterations = 1

    matrices = []
    for i, mouse in enumerate(df.mouse_id.unique()):

        pre_early = np.zeros((iterations, 4, 4))
        pre_late = np.zeros((iterations, 4, 4))
        early_late = np.zeros((iterations, 4, 4))

        mask_pre = df[(df['mouse_id'] == mouse) & (df['period'] == 'pre')]['classes'].iloc[0]
        mask_early = df[(df['mouse_id'] == mouse) & (df['period'] == 'early')]['classes'].iloc[0]
        mask_late = df[(df['mouse_id'] == mouse) & (df['period'] == 'late')]['classes'].iloc[0]

        for it in range(iterations):

            # Shuffle identity masks once per iteration
            if (shuffle is not None) and shuffle:
                rng = np.random.default_rng()
                mask_early = rng.permutation(mask_early)
                mask_late = rng.permutation(mask_late)

            pre_early[it] = func.transition_matrix(mask_pre, mask_early, percent=False)
            early_late[it] = func.transition_matrix(mask_early, mask_late, percent=False)
            pre_late[it] = func.transition_matrix(mask_pre, mask_late, percent=False)

        # Average matrices across shuffles
        if iterations > 1 and return_shuffle_avg:
            pre_early = np.mean(pre_early, axis=0)
            early_late = np.mean(early_late, axis=0)
            pre_late = np.mean(pre_late, axis=0)

        matrices.append(pd.DataFrame([dict(mouse_id=mouse, pre_early=pre_early.squeeze(), pre_late=pre_late.squeeze(),
                                           early_late=early_late.squeeze())]))

    matrices = pd.concat(matrices, ignore_index=True)

    # Make average transition matrices
    avg_trans = {}
    for k, v in groups.items():
        v_1 = [f"{x}_1" for x in v]
        avg_trans[k + '_early'] = np.mean(np.array(list(matrices[matrices['mouse_id'].isin(v_1)]['pre_early'])), axis=0)
        avg_trans[k + '_late'] = np.mean(np.array(list(matrices[matrices['mouse_id'].isin(v_1)]['early_late'])), axis=0)
        avg_trans[k + '_pre_late'] = np.mean(np.array(list(matrices[matrices['mouse_id'].isin(v_1)]['pre_late'])), axis=0)

    def unravel_matrix(mat_early: np.ndarray, mat_late: np.ndarray) -> np.ndarray:
        """
        Unravel transition matrix into the shape that Plotlys Sankey Plots understand:
        A 2D matrix with three rows: First row are the source classes, second row are the target classes,
        third row are the values (how many units go from source to target).
        Classes are split between cell classes and phase.
        Prestroke: ["lost", "non-coding", "unstable", "stable"] -> [0, 1, 2, 3]
        Early Poststroke: ["lost", "non-coding", "unstable", "stable"] -> [4, 5, 6, 7]
        Late Poststroke: ["lost", "non-coding", "unstable", "stable"] -> [8, 9, 10, 11]

        Args:
            mat_early: Transition matrix for pre->early transition. Rows are source classes, columns are target classes.
            mat_late: Transition matrix for early->late transition. Rows are source classes, columns are target classes.

        Returns:
            2D array with shape (3, n_classes*(n_classes*(n_phases-1))) containing data for Sankey plot.
        """
        source_early = [it for sl in [[i] * 4 for i in range(4)] for it in sl]  # "From" class
        target_early = [4, 5, 6, 7] * 4  # "To" classes (shifted by 4)
        source_late = np.array(source_early) + 4
        target_late = np.array(target_early) + 4  # Classes from early->late have different labels (shifted by 4 again)

        early_flat = mat_early.flatten()
        late_flat = mat_late.flatten()

        # Concatenate everything into three rows (source, target, value)
        out = np.array([[*source_early, *source_late], [*target_early, *target_late], [*early_flat, *late_flat]])
        return out

    if directory is not None:
        for k in groups.keys():
            # Export classes for Plotly Sankey Diagram
            sankey_matrix = unravel_matrix(avg_trans[k + '_early'], avg_trans[k + '_late'])
            sankey_matrix = np.round(sankey_matrix)
            np.savetxt(os.path.join(directory, f'{k}.csv'), sankey_matrix, fmt="%d", delimiter=',')
    else:
        if return_group_avg:
            return avg_trans
        else:
            return matrices


def plot_transition_matrix(matrix_list, titles, normalize: True):

    fig, ax = plt.subplots(2, len(titles), layout='constrained', figsize=(10+(5*(len(titles)-2)), 10))
    for i, mat in enumerate(matrix_list):

        if normalize:
            # Normalize matrices row-wise
            mat_early = mat[0] / np.sum(mat[0], axis=1)[..., np.newaxis] * 100
            mat_late = mat[1] / np.sum(mat[1], axis=1)[..., np.newaxis] * 100
        else:
            mat_early = mat[0]
            mat_late = mat[1]

        sns.heatmap(mat_early, ax=ax[0, i], square=True, annot=True, cbar=False, fmt='.3g', vmin=0, vmax=100)
        sns.heatmap(mat_late, ax=ax[1, i], square=True, annot=True, cbar=False, fmt='.3g', vmin=0, vmax=100)
        ax[0, i].set_title(titles[i])
    ax[0, 0].set_xlabel('To Cell Class Early')
    ax[0, 0].set_ylabel('From Cell Class Pre')
    ax[1, 0].set_xlabel('To Cell Class Late')
    ax[1, 0].set_ylabel('From Cell Class Early')


def transition_matrix_for_prism(matrix_df: pd.DataFrame, phase, include_lost=False, norm='rows'):

    # Forward (row-normalization)
    dicts = []
    for _, row in matrix_df.iterrows():
        if include_lost:
            if norm in ['rows', 'forward']:
                mat = row[phase] / np.sum(row[phase], axis=1)[..., np.newaxis] * 100
            elif norm in ['cols', 'backward']:
                mat = row[phase] / np.sum(row[phase], axis=0) * 100
            elif norm in ['all']:
                mat = row[phase] / np.sum(row[phase]) * 100
            elif norm in ['none']:
                mat = row[phase]
            else:
                raise NotImplementedError

            ### Include "lost"
            dicts.append(dict(trans='non-coding > non-coding', mouse_id=int(row['mouse_id'].split('_')[0]), perc=mat[1, 1]))
            dicts.append(dict(trans='non-coding > unstable', mouse_id=int(row['mouse_id'].split('_')[0]), perc=mat[1, 2]))
            dicts.append(dict(trans='non-coding > stable', mouse_id=int(row['mouse_id'].split('_')[0]), perc=mat[1, 3]))
            dicts.append(dict(trans='unstable > non-coding', mouse_id=int(row['mouse_id'].split('_')[0]), perc=mat[2, 1]))
            dicts.append(dict(trans='unstable > unstable', mouse_id=int(row['mouse_id'].split('_')[0]), perc=mat[2, 2]))
            dicts.append(dict(trans='unstable > stable', mouse_id=int(row['mouse_id'].split('_')[0]), perc=mat[2, 3]))
            dicts.append(dict(trans='stable > non-coding', mouse_id=int(row['mouse_id'].split('_')[0]), perc=mat[3, 1]))
            dicts.append(dict(trans='stable > unstable', mouse_id=int(row['mouse_id'].split('_')[0]), perc=mat[3, 2]))
            dicts.append(dict(trans='stable > stable', mouse_id=int(row['mouse_id'].split('_')[0]), perc=mat[3, 3]))

        else:
            if norm in ['rows', 'forward']:
                mat = row[phase][1:, 1:] / np.sum(row[phase][1:, 1:], axis=1)[..., np.newaxis] * 100
            elif norm in ['cols', 'backward']:
                mat = row[phase][1:, 1:] / np.sum(row[phase][1:, 1:], axis=0) * 100
            elif norm in ['all']:
                mat = row[phase][1:, 1:] / np.sum(row[phase][1:, 1:]) * 100
            elif norm in ['none']:
                mat = row[phase]
            else:
                raise NotImplementedError

            ### Exclude "lost"
            dicts.append(dict(trans='non-coding > non-coding', mouse_id=int(row['mouse_id'].split('_')[0]), perc=mat[0, 0]))
            dicts.append(dict(trans='non-coding > unstable', mouse_id=int(row['mouse_id'].split('_')[0]), perc=mat[0, 1]))
            dicts.append(dict(trans='non-coding > stable', mouse_id=int(row['mouse_id'].split('_')[0]), perc=mat[0, 2]))
            dicts.append(dict(trans='unstable > non-coding', mouse_id=int(row['mouse_id'].split('_')[0]), perc=mat[1, 0]))
            dicts.append(dict(trans='unstable > unstable', mouse_id=int(row['mouse_id'].split('_')[0]), perc=mat[1, 1]))
            dicts.append(dict(trans='unstable > stable', mouse_id=int(row['mouse_id'].split('_')[0]), perc=mat[1, 2]))
            dicts.append(dict(trans='stable > non-coding', mouse_id=int(row['mouse_id'].split('_')[0]), perc=mat[2, 0]))
            dicts.append(dict(trans='stable > unstable', mouse_id=int(row['mouse_id'].split('_')[0]), perc=mat[2, 1]))
            dicts.append(dict(trans='stable > stable', mouse_id=int(row['mouse_id'].split('_')[0]), perc=mat[2, 2]))

    df = pd.DataFrame(dicts).pivot(index='trans', columns='mouse_id', values='perc')
    if norm in ['rows', 'forward']:
        df = df.reindex(['non-coding > non-coding', 'non-coding > unstable', 'non-coding > stable', 'unstable > non-coding',
                         'unstable > unstable', 'unstable > stable', 'stable > non-coding', 'stable > unstable', 'stable > stable'])
    elif norm in ['cols', 'backward']:
        df = df.reindex(['non-coding > non-coding', 'unstable > non-coding', 'stable > non-coding',
                         'non-coding > unstable', 'unstable > unstable', 'stable > unstable',
                         'non-coding > stable', 'unstable > stable', 'stable > stable'])
    elif norm in ['all', 'none']:
        df = df.reindex(['non-coding > non-coding', 'non-coding > unstable', 'non-coding > stable', 'unstable > non-coding',
                         'unstable > unstable', 'unstable > stable', 'stable > non-coding', 'stable > unstable', 'stable > stable'])
    # df = df.fillna(0)

    return df




#%%
if __name__ == '__main__':
    spatial_maps = dc.load_data('spat_dff_maps')
    is_pc = dc.load_data('is_pc')

    # To produce CSVs for sankey plots (plotting happens in Jupyter Notebook via Plotly)
    stability_classes = classify_stability(is_pc_list=is_pc, spatial_map_list=spatial_maps, for_prism=False, ignore_lost=True, align_days=True)
    stability_sankey(stability_classes, directory=r'C:\Users\hheise.UZH\Desktop\preprint\class_quantification\sankey_plot')

    # To get class sizes for Prism
    stability_classes = classify_stability(is_pc_list=is_pc, spatial_map_list=spatial_maps, for_prism=True, ignore_lost=True, align_days=True)
    stability_classes.to_clipboard(index=True)


    # Quantify transitions
    trans_matrices = stability_sankey(df=stability_classes, directory=None)
    trans_matrices_rng = stability_sankey(df=stability_classes, directory=None, shuffle=500, return_shuffle_avg=True)

    # Plot transition matrices
    plot_transition_matrix(matrix_list=[[trans_matrices['sham_early'], trans_matrices['sham_late']],
                                        [trans_matrices['no_deficit_early'], trans_matrices['no_deficit_late']],
                                        [trans_matrices['recovery_early'], trans_matrices['recovery_late']],
                                        [trans_matrices['no_recovery_early'], trans_matrices['no_recovery_late']]],
                           titles=['Sham', 'No Deficit', 'Recovery', 'No Recovery'], normalize=True)
    plot_transition_matrix(matrix_list=[[trans_matrices['control_early'], trans_matrices['control_late']],
                                        [trans_matrices['stroke_early'], trans_matrices['stroke_late']]],
                           titles=['No Deficit', 'Stroke'], normalize=False)

    # Export transition matrix for prism
    transition_matrix_for_prism(trans_matrices, phase='pre_early', include_lost=False, norm='forward').to_clipboard(header=True, index=True)
    transition_matrix_for_prism(trans_matrices_rng, phase='early_late', include_lost=False, norm='forward').to_clipboard(header=True, index=True)
    transition_matrix_for_prism(trans_matrices_rng, phase='pre_early', include_lost=False, norm='forward').to_clipboard(header=True, index=True)

# for i in range(len(stability_classes_new)):
#     class_new = stability_classes_new.loc[i, 'classes']
#     class_old = stability_classes_orig.loc[i, 'classes']
#
#     np.sum(class_new - class_old)

    # Find numbers of cells that stay stable PCs for all three phases
    dfs = []
    for mouse in stability_classes.mouse_id.unique():
        classes = np.sum(np.stack(stability_classes.loc[stability_classes.mouse_id == mouse, 'classes']), axis=0)
        dfs.append(pd.DataFrame(index=[int(mouse.split("_")[0])], data=dict(n_stable=np.sum(classes == 9), n_total=len(classes), frac=np.sum(classes == 9)/len(classes)*100)))
    always_stable = pd.concat(dfs)
    always_stable.drop(index=121, inplace=True)

    always_stable.n_stable.sum()
    always_stable.frac.mean()
    always_stable.frac.std()/np.sqrt(len(always_stable.frac))
