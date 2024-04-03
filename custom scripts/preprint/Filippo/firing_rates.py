#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 14/03/2024 13:47
@author: hheise

"""
# script to reproduce figure 1 from Yue Kris Wu's 2020 PNAS paper Homeostatic mechanisms regulate distinct aspects ofcortical circuit dynamics
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import itertools
import scipy as sp
import pandas as pd

import sys

sys.path.append('../')
from preprint.Filippo.utilities import remove_unwanted_mice, avg_over_columns, divide_pre_early_late, df_corr_result, ax_plot_coor_fit, \
    get_group_results_slope_corr, plot_quantiles_with_data, get_group_results_slope_corr


def get_statistic_celltype(firing_rates, place_cells, cellcat=0):
    firing_rate_statistic = {}
    for mouse in firing_rates.keys():
        mouse_rates = {}
        for (day, ratecol), (day, pccol) in zip(firing_rates[mouse].items(), place_cells[mouse].items()):
            try:
                mouse_rates[day] = ratecol.dropna()[pccol.dropna() == cellcat]
            except:
                print(f'mouse {mouse}, day {day}')
        firing_rate_statistic[mouse] = pd.DataFrame(mouse_rates)

    return firing_rate_statistic


def make_boxplots_pre_early_late(fig, axs, statistic_df_list, mouse_groups_coarse,
                                 categories_str_list=['control', 'stroke'],
                                 titlestr_list=[None, None, None]):
    # cb: callable
    # cstring: string of the callable

    for i, titlestr, statistic in zip(range(len(axs)), titlestr_list, statistic_df_list):
        data_arrays = [statistic.loc[group].dropna() for group in mouse_groups_coarse]

        axs[i] = plot_quantiles_with_data(axs[i], categories_str_list,
                                          data_arrays, titlestr=titlestr)

    return fig


def divide_pre_early_late_avg_columns(df, last3_pre=True):
    pre_early_late_division = divide_pre_early_late(df)
    mean_over_columns = [avg_over_columns(pre_early_late_division[0], last3=last3_pre),
                         avg_over_columns(pre_early_late_division[1]),
                         avg_over_columns(pre_early_late_division[2])]
    return mean_over_columns


def get_correlation_pval_fits_subsequent_sessions(rate_df):
    allmice_day_pairs_correlation_pval_fit_slope = {}
    for mouse, df in rate_df.items():
        c = df.columns
        l = len(c)
        mouse_day_pairs_correlation_pval_fit_slope = {}
        for i in range(1, l):
            day_pair = (c[i - 1], c[i])
            t = df[list(day_pair)].dropna()
            nsamples = t.shape[0]
            try:
                corr, pval = sp.stats.pearsonr(t.iloc[:, 0], t.iloc[:, 1])
                regression_result = sp.stats.linregress(t.iloc[:, 0], t.iloc[:, 1])
                slope = regression_result.slope
                intercept = regression_result.intercept
                mouse_day_pairs_correlation_pval_fit_slope[day_pair] = np.array(
                    [corr, pval, slope, intercept, nsamples])
            except:
                mouse_day_pairs_correlation_pval_fit_slope[day_pair] = np.NaN
        allmice_day_pairs_correlation_pval_fit_slope[mouse] = mouse_day_pairs_correlation_pval_fit_slope

    return pd.DataFrame(allmice_day_pairs_correlation_pval_fit_slope).T


def get_pre_early_late_columns_multiindex(df_multiindex):
    cols_prestroke = []
    cols_early_poststroke = []
    cols_late_poststroke = []
    for i in df_multiindex.columns:
        cols_prestroke.append(i[1] < 0)
        cols_early_poststroke.append((i[0] >= 0) and (i[1] < 7))  # this excludes comparison of pre-and poststroke days!
        cols_late_poststroke.append((i[1] >= 7))
    return cols_prestroke, cols_early_poststroke, cols_late_poststroke


def subsequent_sessions_analysis(rate_dict, statnames=['corr', 'pval', 'slope', 'intercept', 'nsamples']):
    subsequent_rates_corr_fit = get_correlation_pval_fits_subsequent_sessions(rate_dict).sort_index(axis=1,
                                                                                                    level=1)  # 0: correlation, 1: pval, 2: fit slope, 3: intercept, 4: nsamples
    separate_rate_statistics = [
        subsequent_rates_corr_fit.applymap(lambda cell: cell[i], na_action='ignore').droplevel(0, axis=1) for i in
        range(5)]
    cols_pre_early_late_multiind = get_pre_early_late_columns_multiindex(subsequent_rates_corr_fit)
    divided_rates_statistics = [[stat.iloc[:, cols] for cols in cols_pre_early_late_multiind] for stat in
                                separate_rate_statistics]
    avg_rate_statistics_pre_early_late = [
        [avg_over_columns(x[0], last3=True).rename(f'pre-{name}'), avg_over_columns(x[1]).rename(f'early-{name}'),
         avg_over_columns(x[2]).rename(f'late-{name}')] for x, name in zip(divided_rates_statistics, statnames)]
    return avg_rate_statistics_pre_early_late


def align_mean_rates_neurons_by_mice_for_scatter_analysis(rates_dict):
    all_mice_pre_early_late_matched_rates = {}
    for mouse, df in rates_dict.items():
        mouse_clean_rates_always_present = df.dropna()
        divided_rates = divide_pre_early_late(mouse_clean_rates_always_present)
        avg_pre_early_late_rates = [avg_over_columns(divided_rates[0], last3=True),
                                    avg_over_columns(divided_rates[1]),
                                    avg_over_columns(divided_rates[2])]
        all_mice_pre_early_late_matched_rates[mouse] = avg_pre_early_late_rates
    return all_mice_pre_early_late_matched_rates


def align_mean_rates_neurons_by_period_for_scatter_analysis(rates_dict):
    all_mice_pre_early_late_matched_rates = {}
    all_mice_pre_early_late_matched_rates['pre'] = {}
    all_mice_pre_early_late_matched_rates['early'] = {}
    all_mice_pre_early_late_matched_rates['late'] = {}

    for mouse, df in rates_dict.items():
        mouse_clean_rates_always_present = df.dropna()
        divided_rates = divide_pre_early_late(mouse_clean_rates_always_present)
        all_mice_pre_early_late_matched_rates['pre'][mouse] = avg_over_columns(divided_rates[0], last3=True)
        all_mice_pre_early_late_matched_rates['early'][mouse] = avg_over_columns(divided_rates[1])
        all_mice_pre_early_late_matched_rates['late'][mouse] = avg_over_columns(divided_rates[2])
    return all_mice_pre_early_late_matched_rates


def ax_plot_coor_fit_for_dict(axes, mousecount, mouse, row, xdict, ydict, xrange=np.array([-0.2, 0.8])):
    # function to plot scatters and make both linear fit and correlation value of the point cloud
    x = xrange
    axes[mousecount][row].scatter(xdict[mouse], ydict[mouse], s=1, marker='x', color='r')
    axes[mousecount][row].set_title(f'Mouse {mouse}')
    axes[mousecount][row].plot(x, x, color='k')
    fit = sp.stats.linregress(xdict[mouse], ydict[mouse])
    axes[mousecount][row].plot(x, x * fit.slope + fit.intercept, color='green')
    try:
        coef = sp.stats.pearsonr(xdict[mouse], ydict[mouse])
        axes[mousecount][row].annotate(f'slope = {fit.slope:.2f}\nr = {coef.statistic:.2f}\np = {coef.pvalue:.4f}',
                                       xy=(2, 4), fontsize=10)
        return fit, coef
    except:
        axes[mousecount][row].annotate(f'slope = {fit.slope:.2f}\ntoo few data points', xy=(2, 4), fontsize=10)
        return fit, np.nan


def pointcloud_analysis_rates(rates_dict, datastr):
    rates_aligned_pre_early_late_by_period = align_mean_rates_neurons_by_period_for_scatter_analysis(rates_dict)
    # make a grid of scatterplots, 3 for every mouse
    nmice = len(rates_dict.keys())  # rows
    nints = 3  # number of columns, 1 plot each: (pre-early, pre-late, early-late)

    fig_scatters, axes = plt.subplots(nmice, nints, figsize=(3.5 * nints, 3 * nmice), sharex=True, sharey=True,
                                      dpi=300)
    pre_early_late_fit_corrcoef = {}

    for mousecount, mouse in enumerate(rates_dict.keys()):

        axes[mousecount][0].set_ylabel('Early post corr')
        try:
            fit_pre_early, corrcoef_pre_early = ax_plot_coor_fit_for_dict(axes, mousecount, mouse, 0,
                                                                          rates_aligned_pre_early_late_by_period['pre'],
                                                                          rates_aligned_pre_early_late_by_period[
                                                                              'early'],
                                                                          xrange=np.array([0, 20]))
            pre_early_late_fit_corrcoef[mouse] = {}
            pre_early_late_fit_corrcoef[mouse]['corr_pre_early'] = corrcoef_pre_early
            pre_early_late_fit_corrcoef[mouse]['fit_pre_early'] = fit_pre_early
        except:
            axes[mousecount, 0].annotate('No data', xy=(2, 2))

        axes[mousecount][1].set_ylabel('Late post corr')
        try:
            fit_pre_late, corrcoef_pre_late = ax_plot_coor_fit_for_dict(axes, mousecount, mouse, 1,
                                                                        rates_aligned_pre_early_late_by_period['early'],
                                                                        rates_aligned_pre_early_late_by_period['late'],
                                                                        xrange=np.array([0, 20]))
            pre_early_late_fit_corrcoef[mouse]['corr_pre_late'] = corrcoef_pre_late
            pre_early_late_fit_corrcoef[mouse]['fit_pre_late'] = fit_pre_late
        except:
            axes[mousecount, 1].annotate('No data', xy=(2, 2))

        axes[mousecount][2].set_ylabel('Late post corr')
        try:
            fit_early_late, corrcoef_early_late = ax_plot_coor_fit_for_dict(axes, mousecount, mouse, 2,
                                                                            rates_aligned_pre_early_late_by_period[
                                                                                'early'],
                                                                            rates_aligned_pre_early_late_by_period[
                                                                                'late'],
                                                                            xrange=np.array([0, 20]))
            pre_early_late_fit_corrcoef[mouse]['corr_early_late'] = corrcoef_early_late
            pre_early_late_fit_corrcoef[mouse]['fit_early_late'] = fit_early_late
        except:
            axes[mousecount, 2].annotate('No data', xy=(2, 2))

    axes[mousecount][0].set_xlabel('Pre corr')
    axes[mousecount][1].set_xlabel('Pre corr')
    axes[mousecount][2].set_xlabel('Early corr')

    fig_scatters.tight_layout(rect=(0.05, 0, 1, 0.97))
    fig_scatters.suptitle(f'Rates of neurons as they transition across periods ({datastr})')
    fig_scatters.savefig(f'{savestr}/grid-pre-early-late-scatter-transitions-rates-{datastr}.png')

    control_slope, control_corr, control_pval = get_group_results_slope_corr(control, pre_early_late_fit_corrcoef)
    stroke_slope, stroke_corr, stroke_pval = get_group_results_slope_corr(stroke, pre_early_late_fit_corrcoef)

    no_recovery_slope, no_recovery_corr, no_recovery_pval = get_group_results_slope_corr(no_recovery,
                                                                                         pre_early_late_fit_corrcoef)
    recovery_slope, recovery_corr, recovery_pval = get_group_results_slope_corr(recovery, pre_early_late_fit_corrcoef)
    sham_slope, sham_corr, sham_pval = get_group_results_slope_corr(sham, pre_early_late_fit_corrcoef)

    # Coarse division
    categories_coarse = ['control', 'stroke']
    categories_fine = ['no recovery', 'recovery', 'sham']

    # slopes
    f_slopes, axs = plt.subplots(2, 3, figsize=(18, 14), dpi=300)

    for i, titlestr in enumerate(['Early poststroke vs prestroke',
                                  'Late poststroke vs prestroke',
                                  'Late poststroke vs early poststroke']):
        data_arrays_corr_coarse = [control_slope[:, i], list(filter(lambda x: x == x, stroke_slope[:, i]))]
        axs[0, i] = plot_quantiles_with_data(axs[0, i], categories_coarse, data_arrays_corr_coarse, titlestr=titlestr)

        data_arrays_corr_fine = [list(filter(lambda x: x == x, no_recovery_slope[:, i])), recovery_slope[:, i],
                                 sham_slope[:, i]]
        axs[1, i] = plot_quantiles_with_data(axs[1, i], categories_fine, data_arrays_corr_fine, titlestr=titlestr)

    axs[0, 0].set_ylabel('Fit slopes')
    axs[1, 0].set_ylabel('Fit slopes')
    f_slopes.tight_layout(rect=(0, 0, 1, 0.95))
    f_slopes.suptitle(f'Boxplots for fit slopes of rates of neurons as they transition across periods ({datastr})')
    f_slopes.savefig(f'{savestr}/slopes-coarse-fine-boxplots-{datastr}.png')
    f_slopes.show()

    # correlations
    f_corr, axs = plt.subplots(2, 3, figsize=(18, 14), dpi=300)

    for i, titlestr in enumerate(['Early poststroke vs prestroke',
                                  'Late poststroke vs prestroke',
                                  'Late poststroke vs early poststroke']):
        data_arrays_corr_coarse = [control_corr[:, i], list(filter(lambda x: x == x, stroke_corr[:, i]))]
        axs[0, i] = plot_quantiles_with_data(axs[0, i], categories_coarse, data_arrays_corr_coarse, titlestr=titlestr)

        data_arrays_corr_fine = [list(filter(lambda x: x == x, no_recovery_corr[:, i])), recovery_corr[:, i],
                                 sham_corr[:, i]]
        axs[1, i] = plot_quantiles_with_data(axs[1, i], categories_fine, data_arrays_corr_fine, titlestr=titlestr)
    axs[0, 0].set_ylabel('Pearson r')
    axs[1, 0].set_ylabel('Pearson r')
    f_corr.tight_layout(rect=(0, 0, 1, 0.95))
    f_corr.suptitle(
        f'Boxplots for pearson correlations of of rates of neurons as they transition across periods ({datastr})')
    f_corr.savefig(f'{savestr}/correlations-coarse-fine-boxplots-{datastr}.png')
    f_corr.show()

    # export fit results and correlations to csv file!
    resdict = {}
    for mouse, res in pre_early_late_fit_corrcoef.items():
        resdict[mouse] = {statname: stat_obj[0] for statname, stat_obj in res.items()}

    res_df = pd.DataFrame(resdict)
    res_df.to_csv(f'{savestr}/fit-slopes-correlations-pointclouds-rates-period-transitions-{datastr}.csv')

    return rates_aligned_pre_early_late_by_period


def removekey(d, key):
    r = dict(d)
    del r[key]
    return r


def write_to_csv(result_series_list_statistics_pre_early_late, savepath):
    res_df = pd.concat([x
                        for y in result_series_list_statistics_pre_early_late
                        for x in y], axis=1)
    res_df.to_csv(savepath)


def binarize_pc_cat(cell):
    if cell > 1:
        return 1
    else:
        return 0


def binarize_tracked_pcs(pc_dict):
    pc_df_dict = {}
    for mouse, df in pc_dict.items():
        pc_df_dict[mouse] = df.applymap(binarize_pc_cat, na_action='ignore')
    return pc_df_dict


def select_unique_period_category(pc_mouse_cats):
    pre_early_late_mousecats = divide_pre_early_late(pc_mouse_cats)
    clean_cat_list_pre_early_late = []
    for df in pre_early_late_mousecats:
        clean_cat_list_pre_early_late.append(df.apply(lambda row: row.mean(skipna=True), axis=1))

    return pd.concat(clean_cat_list_pre_early_late, axis=1).rename({0: 'pre', 1: 'early', 2: 'late'}, axis=1)


def get_cell_id_transitions(place_cell_category_pre_early_late,
                            period_combinations=list(itertools.combinations(['pre', 'early', 'late'], 2)),
                            cell_transitions=list(itertools.product(np.arange(2), np.arange(2)))):
    cell_transitions_dict = {pair: count for count, pair in enumerate(cell_transitions)}

    allperiods_allmice = {}
    for period_combination in period_combinations:
        period_combinations_mouse = {}
        for mouse, period_cells in place_cell_category_pre_early_late.items():
            mouse_period_pairs_category_cells = {}
            t = period_cells[list(period_combination)].dropna()
            for (catpre, catlate), count in cell_transitions_dict.items():
                s = np.logical_and(t[period_combination[0]] == catpre, t[period_combination[1]] == catlate)
                mouse_period_pairs_category_cells[count] = s[s].index

            period_combinations_mouse[mouse] = mouse_period_pairs_category_cells
        allperiods_allmice[period_combination] = period_combinations_mouse
    return allperiods_allmice


def get_rate_statistic_by_transitions(avg_period_firing_rate, place_cell_category_pre_early_late,
                                      rates_compairson_callable_on_rows=lambda row: np.diff(row)[0],
                                      allcel_statistic_callable=lambda series: np.mean(series),
                                      cell_transitions=list(itertools.product(np.arange(2), np.arange(2)))):
    cell_transitions_dict = {pair: count for count, pair in enumerate(cell_transitions)}

    print("The integer transition categories correspond to the following cell type transitions (before, after)")
    for pair, count in cell_transitions_dict.items():
        print(f"Cell transition categories: {pair}: {count}")
    cell_id_transitions = get_cell_id_transitions(place_cell_category_pre_early_late)

    firing_rate_periods_transition_statistics = {}
    for period_pair, transitions in cell_id_transitions.items():
        firing_rate_transition_statistics_mouse = {}

        for mouse, mouse_cell_ids in transitions.items():
            transition_mouse_statistics = {}

            for count in range(4):
                cell_ids = mouse_cell_ids[count]

                tt = avg_period_firing_rate[mouse][list(period_pair)].loc[cell_ids].apply(
                    rates_compairson_callable_on_rows, axis=1)
                transition_mouse_statistics[count] = allcel_statistic_callable(tt)
            firing_rate_transition_statistics_mouse[mouse] = transition_mouse_statistics
        firing_rate_periods_transition_statistics[period_pair] = firing_rate_transition_statistics_mouse

    period_dfs = {
        period_pair: pd.DataFrame(firing_rate_periods_transition_statistics[(period_pair[0], period_pair[1])]).T
        .rename({count: f'{period_pair[0]}-{period_pair[1]} {count}' for count in range(len(cell_transitions))}, axis=1)
        for period_pair in firing_rate_periods_transition_statistics.keys()}

    final_statistic_df = pd.concat(period_dfs.values(), axis=1)
    return final_statistic_df


def get_raw_rate_by_transitions(avg_period_firing_rate, place_cell_category_pre_early_late,
                                      allcel_statistic_callable=lambda series: np.nanmean(series),
                                      cell_transitions=list(itertools.product(np.arange(2), np.arange(2)))):
    cell_transitions_dict = {pair: count for count, pair in enumerate(cell_transitions)}

    print("The integer transition categories correspond to the following cell type transitions (before, after)")
    for pair, count in cell_transitions_dict.items():
        print(f"Cell transition categories: {pair}: {count}")
    cell_id_transitions = get_cell_id_transitions(place_cell_category_pre_early_late)

    firing_rate_periods_transition_statistics = {}
    for period_pair, transitions in cell_id_transitions.items():
        firing_rate_transition_statistics_mouse = {}

        for mouse, mouse_cell_ids in transitions.items():
            transition_mouse_statistics = {}

            for count in range(4):
                cell_ids = mouse_cell_ids[count]
                period_dict={}
                for period in avg_period_firing_rate[mouse]:
                    period_dict[period] = allcel_statistic_callable(avg_period_firing_rate[mouse][period].loc[cell_ids])
                transition_mouse_statistics[count] = period_dict
            firing_rate_transition_statistics_mouse[mouse] = transition_mouse_statistics
        firing_rate_periods_transition_statistics[period_pair] = firing_rate_transition_statistics_mouse

    period_dfs = {
        period_pair: pd.DataFrame(firing_rate_periods_transition_statistics[(period_pair[0], period_pair[1])]).T
        .rename({count: f'{period_pair[0]}-{period_pair[1]} {count}' for count in range(len(cell_transitions))}, axis=1)
        for period_pair in firing_rate_periods_transition_statistics.keys()}

    # Create new DataFrame with proper structure
    reshaped_dfs = {}
    for transition, df in period_dfs.items():
        column_index = pd.MultiIndex.from_product([df.columns, ['pre', 'early', 'late']], names=['transition', 'phase'])
        new_df = pd.concat([pd.DataFrame(df[col].tolist(), columns=column_index, index=df.index) for col in df.columns[:1]], axis=1)

        # Fill the dataframe
        for row_idx, row in df.iterrows():
            for col_idx, col in row.items():
                for phase_idx, value in col.items():
                    new_df.loc[row_idx, (col_idx, phase_idx)] = value

        reshaped_dfs[transition] = new_df

    final_statistic_df = pd.concat(reshaped_dfs.values(), axis=1)
    return final_statistic_df


def get_transition_frequencies(place_cell_category):

    cell_transitions = list(itertools.product(np.arange(2), np.arange(2)))

    # Sort cells by transition
    cell_id_transitions = get_cell_id_transitions(place_cell_category)

    transition_statistics = {}
    for period_pair, transitions in cell_id_transitions.items():
        transition_statistics_mouse = {}
        for mouse_id, mouse_cell_ids in transitions.items():
            transition_statistics_mouse[mouse_id] = {k: len(v) for k, v in mouse_cell_ids.items()}
        transition_statistics[period_pair] = transition_statistics_mouse

    period_dfs = {
        period_pair: pd.DataFrame(transition_statistics[(period_pair[0], period_pair[1])]).T
        .rename({count: f'{period_pair[0]}-{period_pair[1]} {count}' for count in range(len(cell_transitions))}, axis=1)
        for period_pair in transition_statistics.keys()}

    final_statistic_df = pd.concat(period_dfs.values(), axis=1)
    return final_statistic_df



control = [33, 83, 91, 93, 95, 108, 111, 112, 114, 115, 116, 122]

stroke = [41, 63, 69, 85, 86, 89, 90, 110, 113]

no_recovery = [41, 63, 69, 110]
recovery = [85, 86, 89, 90, 113]
sham = [91, 111, 115, 122, 33, 83, 93, 95, 108, 112, 114, 116]

# remove unwanted mice
control.remove(112)
sham.remove(112)

unwanted_mice = [121, 112]

if __name__ == '__main__':
    os.chdir('/home/ga48kuh/NASgroup/labmembers/filippokiessler/wahl-colab')
    scriptname = os.path.basename(__file__)[:-3]

    # savestr = f'code/08012024/tracked-cells/outputs/{scriptname}'
    savestr = r'C:\Users\hheise.UZH\Desktop\preprint\Filippo'
    if not os.path.isdir(savestr):
        os.mkdir(savestr)

    plt.style.use('code/08012024/plotstyle.mplstyle')

    decon_path = r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Data\analysis\Filippo\hendrik-rates\decon-traces-matched-to-pc.pkl'
    pc_path = r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Data\analysis\Filippo\hendrik-rates\place-cell-classes-matched.pkl'

    with open(decon_path, 'rb') as file:
        decon = pickle.load(file)
    with open(pc_path, 'rb') as handle:
        place_cells_tracked = pickle.load(handle)

    place_cells_tracked_binary = binarize_tracked_pcs(place_cells_tracked)


    # Updated PC classes dataframe
    pc_path = r'C:\Users\hheise.UZH\PycharmProjects\Caiman\custom scripts\preprint\Filippo\neural_data\stability_classes.pkl'
    is_pc_path = r'C:\Users\hheise.UZH\PycharmProjects\Caiman\custom scripts\preprint\Filippo\neural_data\is_pc.pkl'
    with open(pc_path, 'rb') as handle:
        place_cells_tracked_new = pickle.load(handle)
    with open(is_pc_path, 'rb') as handle:
        is_pc_tracked = pickle.load(handle)
    # Mask classes DF with is_pc DF
    masked_classes = {}
    for key in is_pc_tracked:
        curr_classes = place_cells_tracked_new[key]
        curr_pc = is_pc_tracked[key]
        masked_classes[key] = curr_classes.mask(curr_pc.isna(), np.nan)

    # remove mice 112 and 121
    decon = remove_unwanted_mice(decon, unwanted_mice)
    place_cells_tracked_binary = remove_unwanted_mice(place_cells_tracked_binary, unwanted_mice)

    # reset indices to ensure consistency of indices with deconvolved traces
    for mouse in place_cells_tracked_binary.keys():
        place_cells_tracked_binary[mouse] = place_cells_tracked_binary[mouse].reset_index(drop=True)

    freq = 30
    firing_rates = {mouse: df.applymap(lambda cell: np.sum(cell) / (len(cell) / freq), na_action='ignore') for mouse, df
                    in decon.items()}
    z_scored_rates = {mouse: df.apply(lambda col: (col - np.mean(col)) / np.std(col)) for mouse, df in
                      firing_rates.items()}  # z-score by session!

    avg_period_firing_rate = {
        mouse: pd.concat(divide_pre_early_late_avg_columns(df), axis=1).rename({0: 'pre', 1: 'early', 2: 'late'},
                                                                               axis=1)
        for mouse, df in z_scored_rates.items()}
    place_cell_category_pre_early_late = {mouse: select_unique_period_category(df)
                                          for mouse, df in place_cells_tracked_binary.items()}

    final_mean_rate_difference_statistic_df = get_rate_statistic_by_transitions(avg_period_firing_rate,
                                                                                place_cell_category_pre_early_late)
    final_mean_rate_difference_statistic_df.to_csv(
        f'{savestr}/mean-zscore-rate-difference-before-after-by-celltype-transitions.csv')

    # New functions by Hendrik
    mean_rates_per_transition_df = get_raw_rate_by_transitions(avg_period_firing_rate,
                                                               place_cell_category_pre_early_late)
    mean_rates_per_transition_df.to_csv(f'{savestr}/mean-zscore-rates-by-celltype-transitions.csv')

    transition_numbers = get_transition_frequencies(place_cell_category_pre_early_late)
    transition_numbers.to_csv(f'{savestr}/celltype-transition-numbers.csv')

    #####################################################################################################
    """ allday_values_nc = {}
    allday_values_pc = {}
    for m in z_scored_rates.values():
        for d in m.keys():
            allday_values_nc[d] = []
            allday_values_pc[d] = []

    f = plt.figure(figsize=(12, 6))
    for (m, nc), (m,pc) in zip(firing_rate_nc.items(), firing_rate_pc.items()):
        for (d, ncol), (d, pcol) in zip(nc.items(), pc.items()):
            ncval = np.median(ncol.dropna())
            pcval = np.median(pcol.dropna())
            plt.scatter(d, ncval, s = 2, color = 'red')
            plt.scatter(d, pcval, s = 2, color = 'blue')
            allday_values_nc[d].append(ncval)
            allday_values_pc[d].append(pcval)

    sorted_days = np.argsort(list(allday_values_pc.keys()))
    plt.plot(np.array(list(allday_values_nc))[sorted_days], np.array([np.nanmedian(c) for c in allday_values_nc.values()])[sorted_days], color = 'red')
    plt.plot(np.array(list(allday_values_pc))[sorted_days], np.array([np.nanmedian(c) for c in allday_values_pc.values()])[sorted_days], color = 'blue')

    plt.legend(handles = [Line2D([0], [0], marker='o', color='red', label='Non coding',
                          markerfacecolor='red', markersize=5, lw = 0),
                          Line2D([0], [0], marker='o', color='blue', label='Place cell',
                          markerfacecolor='blue', markersize=5, lw = 0)])
    plt.title('Median z-scored firing rates over time, by cell type')
    plt.xlabel('Days')
    plt.ylabel('Firing rate [Hz]')
    plt.show()
    plt.savefig(f'{savestr}/median-z-scored-firing-rates.png', dpi = 300)
    plt.close()

    plt.figure()

    ncall = np.hstack([x.values.flatten() for x in firing_rate_nc.values()])
    pcall = np.hstack([x.values.flatten() for x in firing_rate_pc.values()])

    ncall = ncall[~np.logical_or(np.isnan(ncall), np.isinf(ncall))] #mouse 85 has infinite firing rates!
    pcall = pcall[~np.isnan(pcall)]

    plt.hist(ncall, density = True, alpha = 0.5)
    plt.hist(pcall, density = True, alpha = 0.5)



    datalist = [nc_avg_rate, pc_avg_rate]
    length = len(datalist)
    #Make boxplots
    ylabel_list = ['Nc', 'Pc']
    for mouse_grouping, mouse_group_list, division_name in zip([['control', 'stroke'], ['no recovery', 'recovery', 'sham']], [[control, stroke],
                                                                                                                              [no_recovery, recovery, sham]], 
                                                                                                                              ['coarse', 'fine']):
        #coarse division
        fig, axs = plt.subplots(length,3, figsize = (16,18), sharex = True, sharey = True)
        for i in range(length):
            fig = make_boxplots_pre_early_late(fig, axs[i],datalist[i],
                                                  mouse_group_list, categories_str_list = mouse_grouping)

            axs[i,0].set_ylabel(ylabel_list[i])

        axs[0,0].set_title('Prestroke')
        axs[0,1].set_title('Early ps')
        axs[0,2].set_title('Late ps')
        fig.tight_layout(rect = (0,0,1,0.95))
        fig.suptitle(f'Mean z-scored firing rates (mean of medians)')
        fig.savefig(f'{savestr}/{division_name}-mean-of-median-z-scored-firing-rate.png', dpi = 300)


    ##################################################################################################################################
    #try to make figures of scatter plots where one sees the activity change over time of different neurons

    avg_zscore_rate_statistics_pre_early_late = subsequent_sessions_analysis(z_scored_rates)
    write_to_csv(avg_zscore_rate_statistics_pre_early_late, f'{savestr}/z-scored-rates-statistics-subsequent-days-transitions.csv')

    ylabel_list = ['Correlation', 'P-Value', 'Slope', 'Intercept', 'Number of samples']

    #z-scored firing rates
    for mouse_grouping, mouse_group_list, division_name in zip([['control', 'stroke'], ['no recovery', 'recovery', 'sham']], [[control, stroke],
                                                                                                                              [no_recovery, recovery, sham]], 
                                                                                                                              ['coarse', 'fine']):
            #coarse division
        fig, axs = plt.subplots(len(avg_zscore_rate_statistics_pre_early_late),3, figsize = (16,18), sharex = True, sharey = False)
        for i in range(len(avg_zscore_rate_statistics_pre_early_late)):
            fig = make_boxplots_pre_early_late(fig, axs[i],avg_zscore_rate_statistics_pre_early_late[i],
                                                  mouse_group_list, categories_str_list = mouse_grouping)

            axs[i,0].set_ylabel(ylabel_list[i])

        axs[0,0].set_title('Prestroke')
        axs[0,1].set_title('Early ps')
        axs[0,2].set_title('Late ps')
        fig.tight_layout(rect = (0,0,1,0.95))
        fig.suptitle(f'Mean z-scored firing rates (mean of medians)')
        fig.savefig(f'{savestr}/subsequent-session-change-zscored-rate-statistics.png')


    ########################################################################################################################################
    #do the same analysis as in the scatter plots of correlation pairs    
    recovery.remove(85)
    stroke.remove(85)
    z_scored_rates_no85 =  removekey(z_scored_rates, 85) #for this mouse, the dataframe of z-scored rates has no data that is
    #not a NaN and is imaged on all sessions! try z_scored_rates[85].dropna()
    z_scored_rates_pointcloud_data = pointcloud_analysis_rates(z_scored_rates_no85, 'z-scored-rates')
     """
