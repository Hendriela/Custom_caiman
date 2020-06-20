from standard_pipeline import place_cell_pipeline as pipe, performance_check as performance
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from math import ceil, floor
import pickle
from statannot import add_stat_annotation
from glob import glob
from copy import deepcopy
from datetime import datetime as dt
from pathlib import Path

#%% Calculations


def compute_place_cell_ratio(root, filepath=r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\place_cell_ratio.csv',
                             overwrite=False):
    """
    Calculates the ratio of place cells/total active cells for all PCF objects in the root directory tree.
    Results are saved as a csv at filepath.
    :param root: str, directory that holds PCF objects to be analysed
    :param filepath: str, file path of the results CSV
    :param overwrite: bool flag whether an existing CSV table should be extended or overwritten.
    :return:
    """
    file_list = []
    for step in os.walk(root):
        pcf_file = glob(step[0] + '\\pcf_result*')
        if len(pcf_file) > 0:
            file_list.append(max(pcf_file, key=os.path.getmtime))
    print(f'Found {len(file_list)} PCF files. Starting to load data...')

    if os.path.isfile(filepath) and overwrite is False:
        df = pd.read_csv(filepath, sep='\t', index_col=0, parse_dates=['session'])
        extend_df = True
        print('Extending existing CSV file...')
        row_list = []
    else:
        df = pd.DataFrame(index=np.arange(0, len(file_list)),
                          columns=('mouse', 'session', 'n_cells', 'n_place_cells', 'ratio'))
        extend_df = False
        print('Creating new CSV file...')

    date_format = '%Y%m%d'
    for idx, file in enumerate(file_list):
        pipe.progress(idx, len(file_list), status=f'Processing file {idx+1} of {len(file_list)}...')
        # check if the current session is already in the DataFrame
        if extend_df:
            curr_mouse = file.split(sep=os.sep)[-3]
            curr_session = pd.Timestamp(dt.strptime(file.split(sep=os.sep)[-2], date_format).date())
            if len(df.loc[(df['mouse'] == curr_mouse) & (df['session'] == curr_session)]) > 0:
                continue

        pcf = pipe.load_pcf(os.path.dirname(file), os.path.basename(file))

        if extend_df:
            row_list.append(pd.DataFrame({'mouse': pcf.params['mouse'],
                                          'session': dt.strptime(pcf.params['session'], date_format).date(),
                                          'n_cells': pcf.cnmf.estimates.F_dff.shape[0],
                                          'n_place_cells': len(pcf.place_cells),
                                          'ratio': (len(pcf.place_cells)/pcf.cnmf.estimates.F_dff.shape[0])*100}))
        else:
            # Parse data of this session into the dataframe
            df.loc[idx]['mouse'] = pcf.params['mouse']
            df.loc[idx]['session'] = dt.strptime(pcf.params['session'], date_format).date()
            df.loc[idx]['n_cells'] = pcf.cnmf.estimates.F_dff.shape[0]
            df.loc[idx]['n_place_cells'] = len(pcf.place_cells)
            df.loc[idx]['ratio'] = (len(pcf.place_cells)/pcf.cnmf.estimates.F_dff.shape[0])*100

    # give sessions a continuous id for plotting
    df['session_id'] = -1
    for id, session in enumerate(sorted(df['session'].unique())):
        df.loc[df['session'] == session, 'session_id'] = id
    sess_norm = df['session'] - min(df['session'])
    df['sess_norm'] = [x.days for x in sess_norm]

    df.sort_values(by=['mouse', 'session'], inplace=True)   # order rows for mice and session dates
    df.to_csv(filepath, sep='\t')                           # save dataframe as CSV


def load_all_pc_data(root):
    """
    Loads spatial activity map (pcf.bin_avg_activity) and place cell data (pcf.place_cells) of all PCF objects
    in the root directory tree.
    :param root: str, directory that holds all PCF objects to be loaded
    :return: bin_avg_act; np.array with shape (n_neurons, n_bins) holding spatial activity maps of all neurons
             pc; list with length n_place_cells holding data (indices, place fields) of all place cells
    """
    file_list = []
    for step in os.walk(root):
        pcf_files = glob(step[0]+r'\pcf_results*')
        if len(pcf_files) > 0:
            pcf_file = os.path.splitext(os.path.basename(max(pcf_files, key=os.path.getmtime)))[0]
            file_list.append((step[0], pcf_file))
    file_list.reverse()
    print(f'Found {len(file_list)} PCF files. Start loading...')

    bin_avg_act = None
    pc = None

    for file in file_list:

        curr_pcf = pipe.load_pcf(file[0], file[1])

        if bin_avg_act is None:
            bin_avg_act = deepcopy(curr_pcf.bin_avg_activity)
        else:
            idx_offset = bin_avg_act.shape[0]
            try:
                bin_avg_act = np.vstack((bin_avg_act, curr_pcf.bin_avg_activity))
            except ValueError:
                print(f'Couldnt add place cells from {file[0]} because bin number did not add up.')
                continue  # skip the rest of the loop

        if pc is None:
            pc = deepcopy(curr_pcf.place_cells)
        else:
            # Place cell index has to be offset by the amount of cells already in the array
            curr_pc_offset = []
            for i in curr_pcf.place_cells:
                i = list(i)
                i[0] = i[0] + idx_offset
                curr_pc_offset.append(tuple(i))

            pc.extend(curr_pc_offset)

    return bin_avg_act, pc


def compute_spikerate(root, filepath=r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\spikerate.h5',):

#%% Plotting

def plot_pc_ratio_data_separately(column='ratio', scale=1):
    """
    Reads data from the CSV table created by compute_place_cell_ratio() and plots place cell ratio data separately for
    each mouse. Data to be plotted can be number of total active cells, number of place cells or the ratio in %
    (n_pc/n_c * 100).
    :param column: str, column name of data series to be plotted
    :param scale: scale of axis labels
    :return:
    """
    data = pd.read_csv(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\place_cell_ratio.csv',
                       sep='\t', index_col=0, parse_dates=['session'])

    strokes = [18.5, 23.5]
    sns.set()
    sns.set_style('whitegrid')
    grid = sns.FacetGrid(data, col='mouse', col_wrap=3, height=3, aspect=2)
    grid = grid.map(sns.lineplot, 'session_id', 'ratio')

    for stroke in strokes:
        grid = grid.map(plt.axvline, x=stroke, color='r')
    grid.set_axis_labels('session', column)
    sns.set(font_scale=scale)
    plt.tight_layout()


def plot_all_place_cells(bin_avg_act, place_cells, sort='max', norm=True):
    """
    Plots pcolormesh of all place cells. Takes data from load_all_pc_data().
    :param bin_avg_act: bin_avg_act; np.array with shape (n_neurons, n_bins) with spatial activity maps of all neurons
    :param place_cells: list with length n_place_cells holding data (indices, place fields) of all place cells
    :param sort: str; type of sorting. Can be 'max' (maximum dF/F bin) or 'field' (first place field bin)
    :param norm: bool flag if neuronal traces should be normalized (every neuron's trace ranges from 0 to 1)
    :return:
    """
    place_cell_idx = [x[0] for x in place_cells]
    traces = bin_avg_act[place_cell_idx]
    n_neurons = traces.shape[0]

    # Normalize trace array if wanted
    if norm:
        for i in range(traces.shape[0]):
            traces[i] = (traces[i] - np.min(traces[i])) / np.ptp(traces[i])

    # sort neurons after different criteria
    bins = []
    if sort == 'max':
        for i in range(n_neurons):
            bins.append((i, np.argmax(traces[i, :])))
    elif sort == 'field':
        for i in range(n_neurons):
            bins.append((i, place_cells[i][1][0][0]))  # get the first index of the first place field
    else:
        print(f'Cannot understand sorting command {sort}.')
        for i in range(n_neurons):
            bins.append((i, i))
    bins_sorted = sorted(bins, key=lambda tup: tup[1])

    # re-sort traces array
    new_permut = [x[0] for x in bins_sorted]
    new_idx = np.empty_like(new_permut)
    new_idx[new_permut] = np.arange(len(new_permut))
    sort_traces = traces[new_permut, :]

    # Plot data
    fig = plt.figure()
    ax = plt.gca()
    # fig.suptitle(f'Place cells of 101 analysed sessions (from 12 mice)', fontsize=16)

    img = ax.pcolormesh(sort_traces, cmap='plasma')

    cbar = fig.colorbar(img, ax=ax, fraction=0.1, )  # draw color bar
    cbar.set_label(r'normalized $\Delta$F/F', labelpad=-10)
    cbar.set_ticks([0, 1])
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.yaxis.label.set_size(15)

    # set axis labels and tidy up graph
    ax.set_xlabel('VR position bin', fontsize=15)
    ax.set_ylabel('# neuron', fontsize=15)
    ax.tick_params(axis='both', labelsize=12)
    ax.invert_yaxis()



        # set x ticks to VR position, not bin number
    # trace_ax[-1, 0].set_xlim(0, traces.shape[1])
    # x_locs, labels = plt.xticks()
    # plt.xticks(x_locs, (x_locs * self.params['bin_length']).astype(int), fontsize=15)
    # plt.sca(trace_ax[-1, 0])
    # plt.xticks(x_locs, (x_locs * self.params['bin_length']).astype(int), fontsize=15)
