import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import multisession_analysis.pvc_curves as pvc
from scipy.signal import argrelextrema


def ignore_sessions(i, t, d):
    """
    Removes specified sessions from the traces array.
    :param i: list of str, dates of sessions to be ignored in the analysis
    :param t: np.array containing aligned traces with shape (#neurons, #sessions, #bins)
    :param d: dates of aligned sessions (from alignment.columns)
    :return: t_new; t with sessions removed
    :return: d_new; pd.Index object holding dates with sessions removed
    """
    # Remove sessions that should be ignored
    ignore_mask = np.ones(t.shape[1], dtype=bool)
    for sess in i:
        ignore_mask[np.where(d == sess)[0][0]] = False
    t_new = t[:, ignore_mask, :]
    d_new = d[ignore_mask]

    return t_new, d_new


def correlate_activity(traces, alignment, stroke='20200825', ignore=None, title=None, show_coef=True, ax=None, borders=False):
    """
    Correlates activity of individual cells across sessions. Arguments from multi.align_traces().
    :param traces: np.array containing aligned traces with shape (#neurons, #sessions, #bins)
    :param alignment: pd.DataFrame holding respective neuron IDs of unique tracked cells for every session
    :param stroke: str, date of stroke/microsphere injection
    :param ignore: list of str, optional, dates of sessions to be ignored in the analysis
    :param title: str, optional, if given, correlation matrix is plotted with "title" as title
    :param show_coef: bool flag, optional, if correlation coefficients should be shown in the plot
    :return: corr_mat, np.array with shape (#sessions, #sessions) holding avg corrcoefs across all tracked neurons
    """

    def get_corr_matrix(t, d):
        """
        Get average correlation matrix of all sessions across neurons
        :param t: 3D np.array with shape (#neurons, #sessions, #bins), same as "traces" in outer function
        :param d: 1D np.array of all dates, same as "dates" in outer function
        :return:
        """
        mat = np.zeros((len(d), len(d)))  # matrix that saves correlation coefficients of all cells
        n_sess = np.zeros(
            mat.shape)  # cell number per sessions (accounts for nan sessions where cell was not found)
        for cell in range(len(t)):
            # Calculate correlation matrix for each neuron and add it to the total matrix
            curr_mat = np.corrcoef(t[cell])
            # Handle sessions where the current cell was not found (dont add 1 to n_sess, and set corrcoef from nan to 0)
            curr_add = np.ones(n_sess.shape)
            curr_add[np.isnan(curr_mat)] = 0
            curr_mat = np.nan_to_num(curr_mat)
            # Add both arrays to the total matrix
            mat += curr_mat
            n_sess += curr_add

        # Divide by number of cells to get mean correlation
        mat /= n_sess
        return mat

    dates = alignment.columns
    if ignore is not None:
        traces, dates = ignore_sessions(ignore, traces, dates)

    # Find index of stroke session (smallest non-negative date difference)
    date_diff = dates.astype(int) - int(stroke)
    stroke_idx = np.where(date_diff > 0, date_diff, np.inf).argmin()

    corr_mat = get_corr_matrix(traces, dates)

    if title is not None:
        # Plot correlation matrix with seaborn.heatmap
        if ax is None:
            plt.figure(figsize=(10,8))
        else:
            plt.sca(ax)
        d_diff = pd.to_datetime(dates) - pd.to_datetime(stroke)
        sn.heatmap(corr_mat, annot=show_coef, yticklabels=d_diff.days, xticklabels=d_diff.days,
                   cbar=not show_coef, square=True)
        plt.ylim(len(corr_mat), -0.5)
        plt.yticks(rotation=0, fontsize=15)
        plt.xticks(fontsize=15)
        plt.title(title, fontsize=18)
        plt.ylabel('days since lesion', fontsize=18)
        plt.xlabel('days since lesion', fontsize=18)
        if borders:
            plt.axhline(stroke_idx, color='green', linewidth=5)
            plt.axvline(stroke_idx, ymax=0.948, ymin=0, color='green', linewidth=5)
        plt.tight_layout()

    return corr_mat


def split_place_cells(traces, data, alignment):
    """
    Separates traces of all cells into place cells and non place cells. Arguments from multi.align_traces().
    :param traces: 3D np.array containing aligned traces with shape (#neurons, #sessions, #bins)
    :param data: dict with PCF objects of all aligned sessions (dates as keys)
    :param alignment: pd.DataFrame holding respective neuron IDs of unique tracked cells for every session
    :return: 2 np.arrays with same shape as traces, but NaNs for sessions where the cell was not a place cell and v/v.
    """
    pc_traces = traces.copy()
    non_pc_traces = traces.copy()

    for sess_idx, sess in enumerate(alignment.columns[1:]):
        curr_pc = [x[0] for x in data[sess].place_cells]
        for cell_idx, cell in enumerate(alignment[sess]):
            if cell in curr_pc:
                non_pc_traces[cell_idx, sess_idx] = np.nan
            else:
                pc_traces[cell_idx, sess_idx] = np.nan

    return pc_traces, non_pc_traces


def get_pvc(traces, alignment, ignore=None, max_delta_bin=30, stroke='20200825', save=None):

    dates = alignment.columns
    if ignore is not None:
        traces, dates = ignore_sessions(ignore, traces, dates)

    pvc_curves = np.zeros((traces.shape[0], traces.shape[1], max_delta_bin+1)) + np.nan
    pvc_stdev = np.zeros((traces.shape[0], traces.shape[1], max_delta_bin+1)) + np.nan
    # sec_peak = np.zeros((traces.shape[0], traces.shape[1])) TODO: Make this work with individual cells that have small fluctuating local maxima (e.g. M33 cell 21 session 1)
    min_pvc = np.zeros((traces.shape[0], traces.shape[1]))

    for sess in range(traces.shape[1]):
        curr_mat = traces[:, sess]
        curves, std = pvc.pvc_curve_singlecell(curr_mat.T, max_delta_bins=max_delta_bin)
        pvc_curves[:, sess] = curves
        pvc_stdev[:, sess] = std
        min_pvc[:, sess] = np.nanmin(curves[:, :20], axis=1)
        # for cell in range(len(curves)): TODO: see above, find way to get second peak ratio with unstable traces
        #     curve = curves[cell]
        #     try:
        #         sec_peak[cell, sess] = curve[argrelextrema(curve, np.greater)[0][0]] / \
        #                          curve[argrelextrema(curve, np.less)[0][0]]
        #     except IndexError:
        #         sec_peak[cell, sess] = np.nan

    if save is not None:
        d_diff = pd.to_datetime(dates) - pd.to_datetime(stroke)
        np.savetxt(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\cell_alignments\M33_min_pvc.txt',
                   min_pvc, fmt='%.4f', delimiter='\t', header="\t".join(d_diff.days.astype(str)))


def get_single_spikerate(alignment, data, split_pc=False):

    dates = alignment.columns
    if split_pc:
        fr = (np.zeros((len(alignment), len(dates)))+np.nan, np.zeros((len(alignment), len(dates)))+np.nan)
    else:
        fr = np.zeros((len(alignment), len(dates))) + np.nan
    for date_idx, date in enumerate(dates):
        spike_dist = np.nansum(data[date].cnmf.estimates.spikes, axis=1) / (data[date].cnmf.estimates.spikes.shape[1] /
                                                                            data[date].cnmf.params.data['fr'])
        pc_idx = [x[0] for x in data[date].place_cells]
        for cell_idx, cell in enumerate(alignment[date]):
            if int(cell) != -10:
                if split_pc:
                    if int(cell) in pc_idx:
                        fr[0][cell_idx, date_idx] = spike_dist[int(cell)]
                    else:
                        fr[1][cell_idx, date_idx] = spike_dist[int(cell)]
                else:
                    fr[cell_idx, date_idx] = spike_dist[int(cell)]
    return fr
