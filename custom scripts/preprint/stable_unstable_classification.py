#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 22/08/2023 14:08
@author: hheise

"""
import os
import numpy as np
import pandas as pd
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns

from schema import hheise_grouping
from preprint import data_cleaning as dc
from preprint import placecell_heatmap_transition_functions as func


def classify_stability(is_pc_list, spatial_map_list, for_prism=True, ignore_lost=False):

    mice = [next(iter(dic.keys())) for dic in spatial_map_list]

    dfs = []
    for i, mouse in enumerate(mice):
        rel_dates = np.array(is_pc_list[i][mouse][1])
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


def stability_sankey(df: pd.DataFrame, grouping_id: int = 3, return_avg=False,
                     directory: Optional[str] = r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Papers\preprint\class_quantification\sankey_plot'):
    """
    Export a "stability_classes" Dataframe for Plotly`s Sankey diagram.

    Args:
        df: Stability_classes Dataframe, output from classify_stability().
        grouping_id: Parameter ID of the mouse behavioral grouping to be used.
        directory: Directoy where to save the CSV files for the Sankey plot.
    """

    # Load behavioral grouping from Datajoint
    coarse = (hheise_grouping.BehaviorGrouping & f'grouping_id={grouping_id}' & 'cluster="coarse"').get_groups(as_frame=False)
    fine = (hheise_grouping.BehaviorGrouping & f'grouping_id={grouping_id}' & 'cluster="fine"').get_groups(as_frame=False)

    # Combine dicts, adapt names to file system and remove 121
    groups = {'control': coarse['Control'], 'stroke_with121': coarse['Stroke'],
              'stroke': [x for x in coarse['Stroke'] if x != 121],
              'sham': fine['Sham'], 'no_deficit': fine['No Deficit'], 'recovery': fine['Recovery'],
              'no_recovery_with121': fine['No Recovery'], 'no_recovery': [x for x in fine['No Recovery'] if x != 121]}

    # Export classes for Plotly Sankey Diagram
    matrices = []
    for i, mouse in enumerate(df.mouse_id.unique()):
        mask_pre = df[(df['mouse_id'] == mouse) & (df['period'] == 'pre')]['classes'].iloc[0]
        mask_early = df[(df['mouse_id'] == mouse) & (df['period'] == 'early')]['classes'].iloc[0]
        mask_late = df[(df['mouse_id'] == mouse) & (df['period'] == 'late')]['classes'].iloc[0]

        pre_early_trans = func.transition_matrix(mask_pre, mask_early, percent=False)
        early_late_trans = func.transition_matrix(mask_early, mask_late, percent=False)

        matrices.append(pd.DataFrame([dict(mouse_id=mouse, pre_early=pre_early_trans, early_late=early_late_trans)]))
    matrices = pd.concat(matrices, ignore_index=True)

    # Make average transition matrices
    avg_trans = {}
    for k, v in groups.items():
        v_1 = [f"{x}_1" for x in v]
        avg_trans[k + '_early'] = np.mean(np.array(list(matrices[matrices['mouse_id'].isin(v_1)]['pre_early'])), axis=0)
        avg_trans[k + '_late'] = np.mean(np.array(list(matrices[matrices['mouse_id'].isin(v_1)]['early_late'])), axis=0)

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
            sankey_matrix = unravel_matrix(avg_trans[k + '_early'], avg_trans[k + '_late'])
            sankey_matrix = np.round(sankey_matrix)
            np.savetxt(os.path.join(directory, f'{k}.csv'), sankey_matrix, fmt="%d", delimiter=',')
    else:
        if return_avg:
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


def transition_matrix_for_prism(matrix_df: pd.DataFrame, pre_early=True, include_lost=False, norm='rows'):

    # Forward (row-normalization)
    dicts = []
    for _, row in matrix_df.iterrows():
        if include_lost:
            if pre_early:
                if norm in ['rows', 'forward']:
                    mat = row['pre_early'] / np.sum(row['pre_early'], axis=1)[..., np.newaxis] * 100
                elif norm in ['cols', 'backward']:
                    mat = row['pre_early'] / np.sum(row['pre_early'], axis=0) * 100
                elif norm in ['all']:
                    mat = row['pre_early'] / np.sum(row['pre_early']) * 100
                else:
                    raise NotImplementedError

            else:
                if norm in ['rows', 'forward']:
                    mat = row['early_late'] / np.sum(row['early_late'], axis=1)[..., np.newaxis] * 100
                elif norm in ['cols', 'backward']:
                    mat = row['early_late'] / np.sum(row['early_late'], axis=0) * 100
                elif norm in ['all']:
                    mat = row['early_late'] / np.sum(row['early_late']) * 100
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
            if pre_early:
                if norm in ['rows', 'forward']:
                    mat = row['pre_early'][1:, 1:] / np.sum(row['pre_early'][1:, 1:], axis=1)[..., np.newaxis] * 100
                elif norm in ['cols', 'backward']:
                    mat = row['pre_early'][1:, 1:] / np.sum(row['pre_early'][1:, 1:], axis=0) * 100
                elif norm in ['all']:
                    mat = row['pre_early'][1:, 1:] / np.sum(row['pre_early'][1:, 1:]) * 100
                else:
                    raise NotImplementedError
            else:
                if norm in ['rows', 'forward']:
                    mat = row['early_late'][1:, 1:] / np.sum(row['early_late'][1:, 1:], axis=1)[..., np.newaxis] * 100
                elif norm in ['cols', 'backward']:
                    mat = row['early_late'][1:, 1:] / np.sum(row['early_late'][1:, 1:], axis=0) * 100
                elif norm in ['all']:
                    mat = row['early_late'][1:, 1:] / np.sum(row['early_late'][1:, 1:]) * 100
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
    elif norm in ['all']:
        df = df.reindex(['non-coding > non-coding', 'non-coding > unstable', 'non-coding > stable', 'unstable > non-coding',
                         'unstable > unstable', 'unstable > stable', 'stable > non-coding', 'stable > unstable', 'stable > stable'])
    # df = df.fillna(0)

    return df



if __name__ == '__main__':
    spatial_maps = dc.load_data('spat_dff_maps')
    is_pc = dc.load_data('is_pc')

    stability_classes = classify_stability(is_pc_list=is_pc, spatial_map_list=spatial_maps, for_prism=False, ignore_lost=True)
    stability_classes.to_clipboard(index=False)

    stability_sankey(stability_classes, directory=r'E:\user_backup\wahl_folder_backup\Papers\preprint\pc_heatmaps\sankey_plot')

    # Quantify transitions
    trans_matrices = stability_sankey(df=stability_classes, directory=None)

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
    transition_matrix_for_prism(trans_matrices, pre_early=False, include_lost=False, norm='all').to_clipboard(header=False, index=False)
