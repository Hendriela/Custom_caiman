from standard_pipeline import place_cell_pipeline as pipe
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from copy import deepcopy
from datetime import datetime as dt
import multisession_analysis.pvc_curves as pvc
from scipy.signal import argrelextrema

#%% Calculations


def get_simple_data(root, filepath=r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\simple_data.pickle',
                    overwrite=False, session_range=None, norm_range=None, norm_fields=None):
    """
    Calculates simple data points (meaning one datapoint per session/mouse, like place cell ratio, avg spike rate,
    max PVC slope) for all PCF objects in the root tree. Results are saved as a pickle file at filepath.
    :param root: str, directory that holds PCF objects to be analysed
    :param filepath: str, file path of the results pickle object (must include extension).
    :param overwrite: bool flag whether an existing pickle object should be extended or overwritten.
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
        print('Extending existing file...')
        row_list = []
    else:
        df = pd.DataFrame(index=np.arange(0, len(file_list)),
                          columns=('mouse', 'session', 'n_cells', 'n_place_cells', 'ratio', 'mean_spikerate',
                                   'median_spikerate', 'pvc_slope', 'min_pvc', 'sec_peak_ratio', 'spikerate_dist',
                                   'pvc_curve'))
        extend_df = False
        print('Creating new file...')

    date_format = '%Y%m%d'
    for idx, file in enumerate(file_list):
        # check if the current session is already in the DataFrame
        if extend_df:
            curr_mouse = file.split(sep=os.sep)[-3]
            curr_session = file.split(sep=os.sep)[-2]
            if len(df.loc[(df['mouse'] == curr_mouse) & (df['session'] == curr_session)]) > 0:
                continue
        if session_range is not None:
            sess = int(file.split(sep=os.sep)[-2])
            if sess < session_range[0] or sess > session_range[1]:
                continue

        pcf = pipe.load_pcf(os.path.dirname(file), os.path.basename(file))

        pcf.params['mouse'] = pcf.params['root'].split(os.sep)[-2]
        if len(pcf.params['mouse']) > 3:
            pcf.params['mouse'] = pcf.params['mouse'][-3:]
        pcf.params['session'] = pcf.params['root'].split(os.sep)[-1]

        # Get average spike rate in Hz of all neurons
        spike_dist = np.nansum(pcf.cnmf.estimates.spikes, axis=1) / (pcf.cnmf.estimates.spikes.shape[1]/
                                                                     pcf.cnmf.params.data['fr'])
        avg_spike_rate = np.mean(spike_dist)
        median_spike_rate = np.median(spike_dist)

        # Analyse PVC curve of that session (minimum slope and height of second peak)
        try:
            # pvc_curve = pvc.pvc_curve(pcf.bin_avg_activity, max_delta_bins=150)
            curve = np.load(os.path.join(os.path.dirname(file), 'pvc.npy'))
        except FileNotFoundError:
            curve = pvc.pvc_curve(np.transpose(pcf.bin_avg_activity, (1, 0)), plot=False)[0]
        min_slope = -min(np.diff(curve[:20]))
        try:
            second_peak = curve[argrelextrema(curve, np.greater)[0][0]]/curve[argrelextrema(curve, np.less)[0][0]]
        except IndexError:
            second_peak = np.nan

        if extend_df:
            row_list.append(pd.DataFrame({'mouse': pcf.params['mouse'],
                                          'session': pcf.params['session'],
                                          'n_cells': pcf.cnmf.estimates.F_dff.shape[0],
                                          'n_place_cells': len(pcf.place_cells),
                                          'ratio': (len(pcf.place_cells)/pcf.cnmf.estimates.F_dff.shape[0])*100,
                                          'mean_spikerate': avg_spike_rate,
                                          'median_spikerate': median_spike_rate,
                                          'spikerate_dist': spike_dist,
                                          'pvc_curve': curve,
                                          'pvc_slope': min_slope,
                                          'min_pvc': min(curve[:20]),
                                          'sec_peak_ratio': second_peak}))
        else:
            # Parse data of this session into the dataframe
            df.loc[idx]['mouse'] = pcf.params['mouse']
            df.loc[idx]['session'] = pcf.params['session']
            df.loc[idx]['n_cells'] = pcf.cnmf.estimates.F_dff.shape[0]
            df.loc[idx]['n_place_cells'] = len(pcf.place_cells)
            df.loc[idx]['ratio'] = (len(pcf.place_cells)/pcf.cnmf.estimates.F_dff.shape[0])*100
            df.loc[idx]['spikerate_dist'] = spike_dist
            df.loc[idx]['pvc_curve'] = curve
            df.loc[idx]['mean_spikerate'] = avg_spike_rate
            df.loc[idx]['median_spikerate'] = median_spike_rate
            df.loc[idx]['pvc_slope'] = min_slope
            df.loc[idx]['min_pvc'] = min(curve[:20])
            df.loc[idx]['sec_peak_ratio'] = second_peak

    df.dropna(subset=['mouse'], inplace=True)

    # Set correct datatypes for columns
    df.session = df.session.astype(np.int16)
    df.n_cells = df.n_cells.astype(np.int16)
    df.n_place_cells = df.n_place_cells.astype(np.int16)
    df.ratio = df.ratio.astype(np.float64)
    df.mean_spikerate = df.mean_spikerate.astype(np.float64)
    df.median_spikerate = df.median_spikerate.astype(np.float64)
    df.pvc_slope = df.pvc_slope.astype(np.float64)
    df.sec_peak_ratio = df.sec_peak_ratio.astype(np.float64)

    # give sessions a continuous id for plotting
    df['session_id'] = -1
    for id, session in enumerate(sorted(df['session'].unique())):
        df.loc[df['session'] == session, 'session_id'] = id
    df['sess_norm'] = df['session'].astype(int) - min(df['session'].astype(int))

    # assign groups to mice
    stroke = ['M32', 'M40', 'M41']
    df['group'] = np.nan
    for mouse in df.mouse.unique():
        if mouse in stroke:
            df.loc[df.mouse == mouse, 'group'] = 'lesion'
        else:
            df.loc[df.mouse == mouse, 'group'] = 'control'

    # normalize data
    if norm_fields is None:
        norm_fields = ['n_cells', 'n_place_cells', 'ratio', 'mean_spikerate', 'median_spikerate', 'pvc_slope',
                       'min_pvc', 'sec_peak_ratio']

    if norm_range is not None:
        for field in norm_fields:
            # Find indices of correct rows (pre-stroke sessions for the specific animal)
            df[field + '_norm'] = -1.0
            for mouse in df.mouse.unique():
                norm_factor = df.loc[(df.mouse == mouse) &
                                     ((df.session >= norm_range[0]) & (df.session <= norm_range[1])), field].mean()
                df.loc[df.mouse == mouse, field+'_norm'] = df.loc[df.mouse == mouse, field] / norm_factor

    df.sort_values(by=['mouse', 'session'], inplace=True)  # order rows for mice and session dates
    df.to_pickle(filepath)  # save dataframe as pickle

    return df


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


#%% Prism export

def exp_to_prism_mouse_avg(df, field='lick_bin_norm', fname=None, grouping=True,
                           directory=r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing'):

    if grouping:
        # Get maximum number of mice per group
        max_mice_per_group = max([len(df[df.group == group].mouse.unique()) for group in df.group.unique()])

        # Initialize table
        tab = np.zeros((len(df.session_date.unique()), len(df.group.unique())*max_mice_per_group))
        tab[:] = np.nan

        for row, session in enumerate(df.session_date.unique()):
            for group_count, group in enumerate(df.group.unique()):
                # This iterates through all the mice of one group each session and puts avg value into tab
                for mouse_count, mouse in enumerate(df[df.group == group].mouse.unique()):
                    col = group_count*max_mice_per_group+mouse_count
                    tab[row, col] = df.loc[(df.mouse == mouse) & (df.session_date == session), field].mean()

        tab = np.insert(tab, 0, df.sess_norm.unique() + 1, 1)

        if fname is None:
            fname = f'{field}_grouped_mice_avg.txt'
        np.savetxt(fname=os.path.join(directory, fname), X=tab, fmt='%.2f', delimiter='\t',
                   header=f'Groups: {df.group.unique()}. {max_mice_per_group} mice per group.')

    else:
        # Initialize table
        tab = np.zeros((len(df.session_date.unique()), len(df.mouse.unique())))
        tab[:] = np.nan

        for row, session in enumerate(df.session_date.unique()):
            for mouse_count, mouse in enumerate(df.mouse.unique()):
                tab[row, mouse_count] = df.loc[(df.mouse == mouse) & (df.session_date == session), field].mean()

        tab = np.insert(tab, 0, df.session_date.unique(), 1)

        if fname is None:
            fname = f'{field}_grouped_mice_avg.txt'
        np.savetxt(fname=os.path.join(directory, fname), X=tab, fmt='%.2f', delimiter='\t',
                   header=f'Data: {field}. Mean value per session for each mouse: {df.mouse.unique()}.')

def exp_to_prism_single_trials(df, field='lick_bin_norm', group='mouse', fname=None,
                               directory=r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing'):
    """
    Put DataFrame data in a table that can be imported/copied into Prism.
    :param df:
    :param field:
    :param group:
    :param directory:
    :return:
    """
    max_trials = 0
    for unit in df[group].unique():
        for session in df.session_date.unique():
            curr_trials = len(df[(df[group] == unit) & (df.session_date == session)])
            if max_trials < curr_trials:
                max_trials = curr_trials

    tab = np.zeros((len(df.session_date.unique()), len(df[group].unique())*max_trials))
    tab[:] = np.nan

    for row, session in enumerate(df.session_date.unique()):
        group_count = 0
        for unit in df[group].unique():
            curr_data = df.loc[(df[group] == unit) & (df.session_date == session), field]
            tab[row, np.arange(start=group_count*max_trials, stop=group_count*max_trials+len(curr_data))] = curr_data
            group_count += 1

    tab = np.insert(tab, 0, df.session_date.unique(), 1)

    if fname is None:
        fname = f'{field}_grouped_by_{group}_single_trials.txt'
    np.savetxt(fname=os.path.join(directory, fname), X=tab, fmt='%.2f', delimiter='\t',
               header=f'Grouped by {group}: {df[group].unique()}. {max_trials} fields per {group}.')




#%% Plotting

def plot_simple_data_single_mice(df, field, session_range=None, scale=1, stroke_sess=[4.5]):

    sns.set()
    sns.set_style('whitegrid')

    sessions = np.array(np.sort(df['sess_norm'].unique()), dtype=int)
    grid = sns.FacetGrid(df, col='mouse', col_wrap=3, height=3, aspect=2)

    grid.map(sns.lineplot, 'session_id', field)
    grid.set_axis_labels('session', field)

    if session_range is None:
        out = grid.set(xticks=range(len(sessions)), xticklabels=sessions)
    else:
        out = grid.set(xlim=(session_range[0], session_range[1]), xticks=range(len(sessions)), xticklabels=sessions)

    for ax in grid.axes.ravel():
        for sess in stroke_sess:
            ax.axvline(sess, color='r')

    sns.set(font_scale=scale)
    plt.tight_layout()


def plot_simple_data_group_avg(df, field, session_range=None, scale=1, stroke_sess=[4.5]):

    sns.set()
    sns.set_style('whitegrid')

    fig = plt.figure()
    sns.lineplot(x='session_id', y=field, hue='group', data=df)

    for sess in stroke_sess:
        plt.axvline(sess, color='r')

    sns.set(font_scale=scale)
    plt.tight_layout()

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


def plot_all_place_cells(bin_avg_act, place_cells, sort='max', norm=True, cmap='plasma', zone_borders=None):
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



    img = ax.pcolormesh(sort_traces, cmap=cmap)

    cbar = fig.colorbar(img, ax=ax, fraction=0.1)  # draw color bar
    if norm:
        cbar.set_label(r'normalized $\Delta$F/F', labelpad=-10)
        cbar.set_ticks([0, 1])
    else:
        cbar.set_label(r'$\Delta$F/F', labelpad=-5)

    cbar.ax.tick_params(labelsize=20)
    cbar.ax.yaxis.label.set_size(23)

    # Draw RZ locations
    if zone_borders is not None:
        for zone in zone_borders:
            ax.axvspan(zone[0], zone[1], facecolor='orange', alpha=0.3)

    # Set X-Ticks to VR distance instead of bins
    ax.set_xticklabels((ax.get_xticks() * 5).astype(int))

    # set axis labels and tidy up graph
    ax.set_xlabel('VR position', fontsize=30)
    ax.set_ylabel('place cells', fontsize=30)
    ax.tick_params(axis='both', labelsize=18)
    ax.invert_yaxis()

    plt.xticks([])
    plt.yticks([])
    # plt.title('Familiar context (pre-lesion)', fontsize=30, loc='left')
    plt.tight_layout()



        # set x ticks to VR position, not bin number
    # trace_ax[-1, 0].set_xlim(0, traces.shape[1])
    # x_locs, labels = plt.xticks()
    # plt.xticks(x_locs, (x_locs * self.params['bin_length']).astype(int), fontsize=15)
    # plt.sca(trace_ax[-1, 0])
    # plt.xticks(x_locs, (x_locs * self.params['bin_length']).astype(int), fontsize=15)
