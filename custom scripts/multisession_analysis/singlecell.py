import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd

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

    # Remove sessions that should be ignored
    dates = alignment.columns
    if ignore is not None:
        ignore_mask = np.ones(traces.shape[1], dtype=bool)
        for sess in ignore:
            ignore_mask[np.where(dates == sess)[0][0]] = False
        traces = traces[:, ignore_mask, :]
        dates = dates[ignore_mask]

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


def


