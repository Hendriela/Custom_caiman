#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 22/08/2023 14:08
@author: hheise

"""
import os
import numpy as np
import pandas as pd
from typing import Optional, Iterable, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage

from schema import hheise_grouping, hheise_placecell, common_match, hheise_behav
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


def plot_matched_cells_across_sessions_correlation_split(traces: list, is_pc_list: list, plot_sessions: List[int],
                                                         match_matrices: list,
                                                         groups: Tuple[str, str], sort_session: int = 0,
                                                         quantiles: Tuple[float, float] = (0.25, 0.75),
                                                         rel_noncoding_size: int = 2, cross_validation: bool=False,
                                                         normalize: bool = True, across_sessions: bool = False,
                                                         titles: Optional[Iterable] = None, compute_new: bool=False,
                                                         smooth: Optional[int] = None, cmap='turbo'):
    """
    Plot traces of matched neurons across sessions. Neurons are sorted by location of max activity in a given session.
    Args:
        traces: Traces to plot, loaded from data cleaner pickle.
        is_pc_list: Which cells are place cells, loaded from data cleaner pickle.
        plot_sessions: List of days after injection (rel_days) to plot
        groups: List with 2 elements, names of behavioral groups to plot in the upper and lower row
        sort_session: Day after injection where sorting should take place.
        quantiles: Tuple of 2 values giving lower and upper quantile of correlation for stable/unstable split
        rel_noncoding_size: Number of noncoding cells to plot, relative to number of place cells plotted
        normalize: Bool Flag whether the activity should be normalized for each neuron.
        across_sessions: Bool Flag, if normalize=True, whether neuron activity should be normalized across sessions or
                or for each session separately.
        titles: List of titles for each subplot/session.
        smooth: Bool Flag whether the activity should be smoothed, and with which sigma.
        cmap: Color map used to plot the traces.
    """

    # Build arrays
    sort_session_idx = np.where(np.array(plot_sessions) == sort_session)[0][0]

    if compute_new:
        dfs = []
        for mouse_dict, is_pc_dict, match_dict in zip(spatial_maps, is_pc_list, match_matrices):

            # Extract data from dicts, align session days
            mouse_id = list(mouse_dict.keys())[0]
            if mouse_id == '121_1':
                continue
            rel_days = np.array(mouse_dict[mouse_id][1])
            trace = mouse_dict[mouse_id][0]
            is_pc_m = is_pc_dict[mouse_id][0]

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
            rel_days[(np.min(rel_sess) <= rel_sess) & (rel_sess < 1)] = np.arange(np.min(rel_sess), 1)
            first_post_idx = np.where(np.array(plot_sessions) > 0)[0][0]

            # Make mask to only use cells that appear in all sessions to be plotted
            trace_plot = trace[:, np.isin(rel_days, plot_sessions)]
            is_pc_plot = is_pc_m[:, np.isin(rel_days, plot_sessions)]
            cell_mask = np.sum(np.isnan(trace_plot[:, :, 0]), axis=1) == 0
            trace_plot = trace_plot[cell_mask]
            is_pc_plot = is_pc_plot[cell_mask]
            sort_sess_mask_idx = match_dict[mouse_id].iloc[cell_mask, np.isin(rel_days, plot_sessions)].iloc[:, sort_session_idx]

            if trace_plot.shape[1] != len(plot_sessions):
                continue

            # Fetch spatial activity maps from odd and even trials of sorting session separately
            sort_sess_key = common_match.MatchedIndex().string2key(sort_sess_mask_idx.name)

            restriction = [dict(mouse_id=int(mouse_id.split("_")[0]), **sort_sess_key, mask_id=i, place_cell_id=2) for i in sort_sess_mask_idx]
            db_mask_id, spat_sorting = (hheise_placecell.BinnedActivity.ROI & restriction).fetch('mask_id', 'bin_activity')

            # Create a mapping from values to their indices in the unsorted array
            value_to_index = {value: index for index, value in enumerate(sort_sess_mask_idx)}

            # Use the mapping to find indices in the unsorted array for elements in the sorted array
            indices_in_unsorted_array = np.array([value_to_index[value] for value in db_mask_id])

            # Use the index mapping to reorder spat_sorting to match the order in trace_plot
            spat_sorting_sorted = np.stack(spat_sorting[indices_in_unsorted_array])

            # Compute spatial activity map from odd and even trials
            spat_sorting_even = np.mean(spat_sorting_sorted[:, :, ::2], axis=2)
            spat_sorting_odd = np.mean(spat_sorting_sorted[:, :, 1::2], axis=2)

            # Substitute the full spatial map in the sorting session with only odd trial for cross-validation
            # trace_plot[:, sort_session_idx, :] = spat_sorting_odd

            # For each neuron, correlate all session pairs
            for neur_idx, cell in enumerate(trace_plot):
                if smooth is not None:
                    cor_mat = np.corrcoef(ndimage.gaussian_filter1d(cell, sigma=smooth, axis=1))
                else:
                    cor_mat = np.corrcoef(cell)
                cor_mat[np.triu_indices(cor_mat.shape[0], 0)] = np.nan
                dfs.append(pd.DataFrame([dict(mouse_id=int(mouse_id.split("_")[0]), session_str=sort_sess_mask_idx.name,
                                              mask_id=int(sort_sess_mask_idx.iloc[neur_idx]),
                                              is_pc=int(is_pc_plot[neur_idx, sort_session_idx]),
                                              pre_cor=np.tanh(np.nanmean(np.arctanh(cor_mat[:first_post_idx]))),
                                              post_cor=np.tanh(np.nanmean(np.arctanh(cor_mat[first_post_idx:]))),
                                              all_cor=np.tanh(np.nanmean(np.arctanh(cor_mat))), traces=cell,
                                              spat_sorting_even=spat_sorting_even[neur_idx],
                                              spat_sorting_odd=spat_sorting_odd[neur_idx])]))
        df = pd.concat(dfs, ignore_index=True)

        df.to_pickle(r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Papers\preprint\class_quantification\example_plot_data_smooth.pickle')

    elif smooth is not None:
        df = pd.read_pickle(r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Papers\preprint\class_quantification\example_plot_data_smooth.pickle')
    elif cross_validation:
        df = pd.read_pickle(r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Papers\preprint\class_quantification\example_plot_data_crossval.pickle')
    else:
        df = pd.read_pickle(r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Papers\preprint\class_quantification\example_plot_data.pickle')

    # Enter groups
    coarse = (hheise_grouping.BehaviorGrouping & 'grouping_id = 4' & 'cluster = "coarse"').get_groups()
    fine = (hheise_grouping.BehaviorGrouping & 'grouping_id = 4' & 'cluster = "fine"').get_groups()
    df = df.merge(coarse, how='left', on='mouse_id').rename(columns={'group': 'coarse'})
    df = df.merge(fine, how='left', on='mouse_id').rename(columns={'group': 'fine'})

    zones = (hheise_behav.CorridorPattern() & 'pattern="training"').rescale_borders(80)

    # Create plot
    fig, ax = plt.subplots(nrows=len(groups), ncols=len(plot_sessions), sharex='all', sharey='row',
                           layout='constrained')

    # For each group, sort and split cells, and plot traces
    for j, group in enumerate(groups):
        if group in coarse.group.values:
            df_filt = df[df.coarse == group]
        elif group in fine.group.values:
            df_filt = df[df.fine == group]
        else:
            raise IndexError(f'Group {group} not found.')

        # if cross_validation:
        #     all_traces = list(df_filt.loc[:, "traces"].copy())
        #     all_even = list(df_filt.loc[:, "spat_sorting_even"].copy())
        #     all_new_traces = []
        #     for all_trace, all_eve in zip(all_traces, all_even):
        #         all_trace = all_trace.copy()
        #         all_trace[sort_session_idx] = all_eve
        #         all_new_traces.append(all_trace)
        #     df_filt['traces'] = all_new_traces

        # Get 25th and 75th quantile of place cell correlation
        lower_quant = np.quantile(df_filt[df_filt.is_pc == 1]['all_cor'], quantiles[0])
        upper_quant = np.quantile(df_filt[df_filt.is_pc == 1]['all_cor'], quantiles[1])

        stable_df = df_filt[(df_filt.is_pc == 1) & (df_filt.all_cor > upper_quant)]
        unstable_df = df_filt[(df_filt.is_pc == 1) & (df_filt.all_cor < lower_quant)]

        # Randomly select subset of noncoding cells
        idx = np.random.choice(np.arange((df_filt.is_pc == 0).sum()), len(stable_df['traces'])*2*rel_noncoding_size, replace=False)
        noncoding_traces = np.stack(df_filt[(df_filt.is_pc == 0)]['traces'].values)[idx]

        # Sort traces based on sort_session
        def sort_traces(key_trace, sort_trace):
            sort_key = [(i, np.argmax(trace)) for i, trace in enumerate(key_trace)]
            sort_key = [y[0] for y in sorted(sort_key, key=lambda x: x[1])]
            return sort_trace[sort_key]

        # Sort via cross-validation
        if cross_validation:
            stable_traces_sort = sort_traces(np.stack(stable_df['spat_sorting_even'].values), np.stack(stable_df['traces'].values))
            unstable_traces_sort = sort_traces(np.stack(unstable_df['spat_sorting_even'].values), np.stack(unstable_df['traces'].values))
            noncoding_traces_sort = sort_traces(np.stack(df_filt[(df_filt.is_pc == 0)]['spat_sorting_even'].values)[idx],
                                                noncoding_traces)
        else:
            # Sort without cross-validation
            stable_traces_sort = sort_traces(np.stack(stable_df['traces'].values)[:, sort_session_idx], np.stack(stable_df['traces'].values))
            unstable_traces_sort = sort_traces(np.stack(unstable_df['traces'].values)[:, sort_session_idx], np.stack(unstable_df['traces'].values))
            noncoding_traces_sort = sort_traces(np.stack(df_filt[(df_filt.is_pc == 0)]['traces'].values)[idx][:, sort_session_idx],
                                                noncoding_traces)

        # Combine arrays with an empty row
        data_arrays = [stable_traces_sort, unstable_traces_sort, noncoding_traces_sort]
        empty_row = np.zeros((1, *data_arrays[0].shape[1:])) * np.nan
        stacked_data = [np.vstack([x, empty_row]) for x in data_arrays[:-1]]
        stacked_data.append(data_arrays[-1])
        stacked_data = np.vstack(stacked_data)

        vmin = np.nanmin(stacked_data)
        vmax = np.nanmax(stacked_data)

        # Plot traces for each day as heatmaps
        for i in range(stacked_data.shape[1]):

            # sns.heatmap(stacked_data[:, i], ax=ax[j, i], cbar=False, cmap='turbo')
            sns.heatmap(ndimage.gaussian_filter1d(stacked_data[:, i], axis=1, sigma=1), ax=ax[j, i], cbar=False, cmap='turbo')

            for z in zones:
                ax[j, i].axvline(z[0], linestyle='--', c='green')
                ax[j, i].axvline(z[1], linestyle='--', c='green')

    # Normalize activity neuron-wise for each sessions
    if normalize:
        if across_sessions:
            sess_squeeze = np.reshape(traces_sort, (traces_sort.shape[0], traces_sort.shape[1] * traces_sort.shape[2]))
            neur_max = np.nanmax(sess_squeeze, axis=1)
            neur_min = np.nanmin(sess_squeeze, axis=1)
            to_plot = (traces_sort - neur_min[:, None, None]) / (neur_max[:, None, None] - neur_min[:, None, None])
        else:
            traces_norm = []
            for i in range(traces_sort.shape[1]):
                neur_sess_max = np.nanmax(traces_sort[:, i, :], axis=1)
                neur_sess_min = np.nanmin(traces_sort[:, i, :], axis=1)
                traces_norm.append((traces_sort[:, i, :] - neur_sess_min[:, None]) /
                                   (neur_sess_max[:, None] - neur_sess_min[:, None]))
            to_plot = np.stack(traces_norm, axis=1)
        vmax = 1
    else:
        to_plot = traces_sort
        vmax = None

    fig, axes = plt.subplots(nrows=1, ncols=traces_sort.shape[1], figsize=(15, 7.3))
    for idx, ax in enumerate(axes):
        sns.heatmap(to_plot[:, idx, :], cmap=cmap, ax=ax, cbar=False, vmax=vmax)

        # Formatting
        if idx != 0:
            ax.set_yticks([])
        else:
            ax.set_ylabel('Cell no.', fontsize=15, labelpad=-3)

        ax.set_xticks([])
        ax.set_xlabel('Track position [m]', fontsize=15)
        # ax.set_title(f'{traces_sort.shape[1]-idx}d prestroke')

        if titles is not None:
            ax.set_title(titles[idx])

    plt.tight_layout()

    return fig


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

    # Plot matched cells across days, split into groups
    match_matrix = dc.load_data('match_matrix')
    plot_matched_cells_across_sessions_correlation_split(traces=spatial_maps, is_pc_list=is_pc, match_matrices=match_matrix,
                                                         plot_sessions=[-2, -1, 0, 3, 6, 12, 15], groups=("Stroke", "Control"))

