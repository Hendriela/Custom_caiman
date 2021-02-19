import os
import numpy as np
import seaborn as sns
from datetime import datetime as dt
from datetime import timedelta as delta
from itertools import compress
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from math import ceil
from copy import deepcopy
import re


def multi_mouse_performance(mouse_dir_list, novel, precise_duration=False, separate_zones=False, date_0='0'):
    if not precise_duration:
        plt.figure(figsize=(15, 8))
        if novel:
            plt.title(f'Trial duration in novel corridor')
        else:
            plt.title(f'Trial duration in training corridor')

        mice_dates = []
        mice_licks = []
        mice_stops = []
        for i in range(len(mouse_dir_list)):
            # get the behavioral data for the current mouse
            if type(date_0) == list:
                if len(mouse_dir_list) == len(date_0):
                    dates, licks, stops = load_performance_data(mouse_dir_list[i], novel, norm_date=date_0[i])
                else:
                    print('If a list, date_0 has to be the same length as mouse_dir_list!')
            elif date_0 is None or date_0 == '0':
                dates, licks, stops = load_performance_data(mouse_dir_list[i], novel, norm_date='0')
            else:
                print('Date_0 input not understood.')
            """
            # filter out sessions with missing data and get a data-specific date list for x-axis labelling
            dates_speed = []
            dates_zones = []
            filter_zones = []
            filter_speed = []
            for j in range(len(dates)):
                if type(speed[j]) != np.int32 and type(speed[j]) != int:
                    dates_speed.append(dates[j])
                    filter_speed.append(speed[j])
                if type(zones[j]) != np.int32 and type(zones[j]) != int:
                    dates_zones.append(dates[j])
                    filter_zones.append(zones[j])
            """
            mice_dates.append(dates)
            mice_licks.append(licks)
            mice_stops.append(stops)

        # save data
        data = pd.DataFrame({
            'Mouse': ['M18', 'M22'],
            'Date': mice_dates,
            'Licks': mice_licks,
            'Stops': mice_stops})
        data.to_pickle(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\all_mice_licking_stopping_novel_nostroke')

        # recover data
        data = pd.read_pickle(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\all_mice_licking_stopping')
        mice_dates = list(data['Date'])
        mice_licks = list(data['Licks'])
        mice_stops = list(data['Stops'])

        mice_licks_avg = [[], [], [], []]
        mice_stops_avg = [[], [], [], []]

        # get mean of sessions
        for mouse in range(len(mice_licks)):
            for session in mice_licks[mouse]:
                mice_licks_avg[mouse].append(np.nanmean(session))
            for session in mice_stops[mouse]:
                mice_stops_avg[mouse].append(np.nanmean(session))

        # combine all dates to have a global list of sessions
        glob_dates = np.unique([item for sublist in mice_dates for item in sublist])
        # insert None for sessions that did not have data
        for i in range(len(mice_licks)):
            for j in range(len(glob_dates)):
                try:
                    if glob_dates[j] != mice_dates[i][j]:
                        mice_dates[i].insert(j, None)
                        mice_licks_avg[i].insert(j, None)
                        mice_stops_avg[i].insert(j, None)
                except IndexError:
                    mice_dates[i].append(None)
                    mice_licks_avg[i].append(None)
                    mice_stops_avg[i].append(None)


        # make mask for plotting
        lick_mask = []
        stop_mask = []
        for i in range(len(mice_licks_avg)):
            mice_licks_avg[i] = np.array(mice_licks_avg[i]).astype(np.double)
            lick_mask.append(np.isfinite(mice_licks_avg[i]))
            mice_stops_avg[i] = np.array(mice_stops_avg[i]).astype(np.double)
            stop_mask.append(np.isfinite(mice_stops_avg[i]))

        # mouse list for legend labels
        labels = ['M18', 'M22']


        # plot licking and stopping data
        fig, ax = plt.subplots(1, 2, sharey=True, figsize=(15, 4))
        fig.suptitle('Behavioral performance of all mice in novel corridor')
        # plot licking data
        ax[0].set_title('Lick ratio')
        ax[1].set_title('Stop ratio')
        ax[0].set_xlabel('days after stroke')
        ax[1].set_xlabel('days after stroke')
        for i in range(len(mice_dates)):
            ax[0].plot(glob_dates[lick_mask[i]], mice_licks_avg[i][lick_mask[i]], linestyle='-', label=labels[i])
            ax[1].plot(glob_dates[stop_mask[i]], mice_stops_avg[i][stop_mask[i]], linestyle='-', label=labels[i])
        ax[0].legend()
        ax[1].legend()
        fig.tight_layout()

        data = pd.DataFrame({
            'Mouse': ['M18', 'M19', 'M22', 'M25'],
            'Date': mice_dates,
            'Licks': mice_licks,
            'Stops': mice_stops})

        data_melt = pd.melt(data, id_vars=['Date'])

        # plot data of the current mouse
        sns.pointplot(x=data['Date'], y=['Speed'], data=filter_speed, label=mouse_dir_list[i][-3])
        sns.pointplot(data=data['Speed'])


def single_mouse_performance(data, novel=False, precise=False, separate_zones=False, date_0='0'):
    """
    :param days: list, labels for each session
    :param licking: list of lists, one list per session, holds licking performance for each trial for that session
    :param stopping: list of lists, one list per session, holds stopping performance for each trial for that session
    :param novel: bool, whether novel or novel corridor sessions should be analysed
    :param precise: bool, whether data should be shown precise (boxplot) or only mean (pointplot)
    :param date_0: str, where the session dates should be normalized. 'None' = no normalization (dates labelled with
                    session folder name), '0' = normalization to the first session, str in format 'YYYYMMDD' = norm. to
                    the session of the respective day (sessions before labelled with negative numbers). If not None or
                    '0', date_0 has to be a folder name of a session of this mouse.
    """

    # separate figures for licking and stopping
    # if precise:
    #     # plot licking
    #     fig_lick, ax_lick = plt.figure(figsize=(15, 8))
    #     if novel:
    #         fig_lick.title(f'{root[-3:]}: Lick ratio in novel corridor')
    #     else:
    #         fig_lick.title(f'{root[-3:]}: Lick ratio in training corridor')
    #     sns.boxplot(ax=ax_lick, data=licking)
    #     set_xlabels(ax_lick, days, date_0)
    #     ax_lick.set_ylabel('Lick ratio')
    #     # plot stopping
    #     fig_stop, ax_stop = plt.figure(figsize=(15, 8))
    #     if novel:
    #         fig_stop.title(f'{root[-3:]}: Stop ratio in novel corridor')
    #     else:
    #         fig_stop.title(f'{root[-3:]}: Stop ratio in training corridor')
    #     sns.boxplot(ax=ax_stop, data=licking)
    #     set_xlabels(ax_stop, days, date_0)
    #     ax_stop.set_ylabel('Stop ratio')
    # one plot for licking and stopping
    # else:
    fig = plt.figure(figsize=(15, 8))
    lick_line = sns.pointplot(data=licking)
    lick_line.set_label('Licks')
    stop_line = sns.pointplot(data=stopping, color='red')
    stop_line.set_label('Stops')
        # TODO: make labels work


def set_xlabels(axis, x_labels, date_0):
    adapt_xlabels = []
    locs, labels = plt.xticks()
    for loc in locs:
        if loc >= 0:
            try:
                adapt_xlabels.append(x_labels[int(loc)])
            except IndexError:
                adapt_xlabels.append('nan')
        else:
            adapt_xlabels.append('nan')

    if date_0 is None:
        axis.set_xticks(x_labels)
        axis.set_xticklabels(adapt_xlabels, rotation=45)
        out = axis.set_xlabel('session dates')
    elif date_0 == '0':
        axis.set_xticklabels(adapt_xlabels)
        out = axis.set_xlabel('days')
    else:
        axis.set_xticklabels(adapt_xlabels)
        out = axis.set_xlabel('days after stroke')


def load_performance_data(roots, norm_date, stroke=None, ignore=None):
    """
    Collects licking and stopping data from merged_behavior.txt files for a list of mice.
    :param roots: str, path to the batch folder (which contains folders for each mouse)
    :param norm_date: date where session dates should be normalized (one entry per mouse or single entry for all mice).
                        'None' = no normalization (dates labelled with session folder name),
                        '0' = normalization to the first session,
                        str in format 'YYYYMMDD' = norm. to the session of the respective day (neg for previous days).
    :param ignore: optional, list of strings of mice IDs that should be ignored during plotting
    """
    data = []
    for root in roots:
        for step in os.walk(root):
            # Find performance
            if 'performance.txt' in step[2] and 'validation' not in step[0]:
                # load data of the current session and add it to the global list as a pd.Series
                file_path = os.path.join(step[0], 'performance.txt')
                sess_data = np.loadtxt(file_path)
                sess_data = np.nan_to_num(sess_data)
                if len(sess_data.shape) == 1:
                    lick = [sess_data[0]]
                    lick_bin = [sess_data[1]]
                    stop = [sess_data[2]]
                    # stop = [sess_data[1]]
                elif len(sess_data.shape) > 1:
                    lick = sess_data[:, 0]
                    lick_bin = sess_data[:, 1]
                    stop = sess_data[:, 2]
                    # stop = sess_data[:, 1]

                else:
                    raise ValueError(f'No data found in {file_path}.')

                sess = step[0].split(os.path.sep)[-1]
                mouse = step[0].split(os.path.sep)[-2]
                # store data of this session and this mouse in a temporary DataFrame which is added to the global list
                if (ignore is None) or (ignore is not None and mouse not in ignore):
                    data.append(pd.DataFrame({'licking': lick, 'stopping': stop, 'licking_binned': lick_bin,
                                              'mouse': mouse, 'session_date': sess,
                                              'novel_corr': is_session_novel(step[0])}))
                # data.append(pd.DataFrame({'licking': lick, 'stopping': stop, 'mouse': mouse, 'session_date': sess}))

    df = pd.concat(data, ignore_index=True)

    # give sessions a continuous id for plotting
    all_sess = sorted(df['session_date'].unique())
    count = 0
    df['sess_id'] = -1
    for session in all_sess:
        df.loc[df['session_date'] == session, 'sess_id'] = count
        count += 1

    # normalize days if necessary # todo: fix with new DF and dict structure
    if type(norm_date) == dict:
        for key in norm_date:
            if norm_date[key] is not None:
                df_mask = df['mouse'] == key
                session_norm = normalize_dates(df['mouse'] == key, norm_date[key])
            else:
                session_norm = sess
    elif type(norm_date) == str:
        session_norm = normalize_dates(list(df['session_date']), norm_date)
    else:
        session_norm = df['session_date']

    df['sess_norm'] = session_norm

    # Set group of mice
    df['group'] = 'unknown'
    if stroke is not None:
        df.loc[df.mouse.isin(stroke), 'group'] = 'stroke'
        df.loc[~df.mouse.isin(stroke), 'group'] = 'control'

    # Calculate cumulative performance
    for mouse in df['mouse'].unique():
        # Set first session (copy of normal performance)
        df.loc[(df['sess_norm'] <= 0) & (df['mouse'] == mouse), 'cum_perf'] = \
            df.loc[(df['sess_norm'] <= 0) & (df['mouse'] == mouse), 'licking']
        prev_cum = df.loc[(df['sess_norm'] <= 0) & (df['mouse'] == mouse), 'cum_perf'].mean()

        for sess in range(1, int(df['sess_norm'].max())+1):
            if np.any(sess == df.loc[df['mouse'] == mouse, 'sess_norm']):
                df.loc[(df['sess_norm'] == sess) & (df['mouse'] == mouse), 'cum_perf'] = \
                    df.loc[(df['sess_norm'] == sess) & (df['mouse'] == mouse), 'licking'] + prev_cum
                prev_cum = df.loc[(df['sess_norm'] == sess) & (df['mouse'] == mouse), 'cum_perf'].mean()

    return df


def get_cum_performance(dataset):

    df = dataset.assign(cum_perf=[0]*len(dataset))

    for mouse in df['mouse'].unique():
        # Set first session (copy of normal performance)
        df.loc[(df['sess_norm'] <= 0) & (df['mouse'] == mouse), 'cum_perf'] = \
            df.loc[(df['sess_norm'] <= 0) & (df['mouse'] == mouse), 'licking']
        prev_cum = df.loc[(df['sess_norm'] <= 0) & (df['mouse'] == mouse), 'cum_perf'].mean()

        for sess in range(1, int(df['sess_norm'].max())+1):
            if np.any(sess == df.loc[df['mouse'] == mouse, 'sess_norm']):
                df.loc[(df['sess_norm'] == sess) & (df['mouse'] == mouse), 'cum_perf'] = \
                    df.loc[(df['sess_norm'] == sess) & (df['mouse'] == mouse), 'licking'] + prev_cum
                prev_cum = df.loc[(df['sess_norm'] == sess) & (df['mouse'] == mouse), 'cum_perf'].mean()

    return df


def save_multi_performance(path, overwrite=False):
    """
    Wrapper function for save_performance_data that goes through folders and looks for sessions that have no
    performance.txt yet. Session folders are determined by the presence of the LOG.txt file.
    :param path: Top-level directory where subdirectories are searched.
    :param path: bool flag whether to overwrite existing performance.txt files.
    """
    print(f'Computing performance of session...')
    for (dirpath, dirnames, filenames) in os.walk(path):
        if len([x for x in filenames if 'TDT LOG' in x]) == 1:
            if 'performance.txt' not in filenames or overwrite:
                print(f'\t{dirpath}')
                save_performance_data(dirpath)
    print('Everything processed!')


def save_performance_data(session, validation=False, use_valve=False):
    """
    Calculates and saves licking and stopping ratios of a session.
    :param session: str, path to the session folder that holds behavioral txt files
    :param validation: bool flag whether to use RZ positions of validation trials (shifted RZs in training corridor)
    :param use_valve:
    :return:
    """

    # Hardcoded validation sessions for Batch 3
    if session[-8:] == '20200614' or session[-8:] == '20200616':
        validation = True

    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split('(\d+)', text)]

    # Get the list of all behavior files in the session folder
    file_list = list()
    for (dirpath, dirnames, filenames) in os.walk(session):
        if len(glob(dirpath + '\\*.tif')) > 0:
            # Trial files named "file" or "wave" have valid behavior, other names not (e.g. "nolick")
            if len(glob(dirpath + '\\file_*.tif')) + len(glob(dirpath + '\\wave_*.tif')) > 0:
                file_list += glob(dirpath + '\\merged_behavior*.txt')
        else:
            file_list += glob(dirpath + '\\merged_behavior*.txt')
    file_list.sort(key=natural_keys)

    if len(file_list) > 0:
        session_performance = np.zeros((len(file_list), 3))
        for i, file in enumerate(file_list):
            if validation and i > 4:
                session_performance[i, 0], session_performance[i, 1],  session_performance[i, 2] = \
                    extract_performance_from_merged(pd.read_csv(file, sep='\t'), novel=is_session_novel(session),
                                                    valid=validation, buffer=0, use_reward=use_valve)
            else:
                session_performance[i, 0], session_performance[i, 1],  session_performance[i, 2] = \
                    extract_performance_from_merged(pd.read_csv(file, sep='\t'), novel=is_session_novel(session),
                                                    buffer=2, use_reward=use_valve)

        file_path = os.path.join(session, f'performance.txt')
        np.savetxt(file_path, session_performance, delimiter='\t',  fmt=['%.4f', '%.4f', '%.4f'],
                   header='Licking\tBinned Licking\tStopping')


def is_session_novel(path):
    # check whether the current session is in the novel corridor
    context = None
    if len(glob(path + '\\TDT LOG*')) > 0:
        log_path = os.path.join(path, glob(path + '\\TDT LOG*')[0])
        with open(log_path, 'r') as log:
            lines = log.readlines()
            for curr_line in lines:
                line = curr_line.split('\t')
                if 'VR enter Reward Zone:' in line[-1]:
                    if int(np.round(float(line[-1][-5:-1]))) == 6:
                        context = 'training'
                    elif int(np.round(float(line[-1][-5:-1]))) == 9:
                        context = 'novel'
    else:
        context = 'training'  # if there is no log file (in first trials), its 'training' by default

    if context == 'training':
        return False
    elif context == 'novel':
        return True
    else:
        print(f'Could not determine context in session {path}!\n')


def get_binned_licking(data, bin_size=2, normalized=True):
    """
    Extracts behavior data from one merged_behavior.txt file (acquired through behavior_import.py).
    :param data: np.array of the merged_behavior*.txt file
    :param bin_size: int, bin size in VR units for binned licking performance analysis
    :param normalized: bool flag whether lick counts should be returned normalized (sum of bins = 1)
    :return hist: 1D np.array with length 120/bin_size holding (normalized) binned lick counts per position bin
    """
    # select only time point where the mouse licked
    lick_only = data[np.where(data[:, 2] == 1)]

    if lick_only.shape[0] == 0:
        hist = np.zeros(int(120/bin_size))
    else:
        # split licking track into individual licks and get VR position
        diff = np.round(np.diff(lick_only[:, 0]) * 10000).astype(int)  # get an array of time differences
        licks = np.split(lick_only, np.where(diff > 5)[0] + 1)  # split at points where difference > 0.5 ms (sample gap)
        licks = [i for i in licks if i.shape[0] <= 10000]      # only keep licks shorter than 5 seconds (10,000 samples)
        lick_pos = [x[0, 1] for x in licks]                    # get VR position when each lick begins

        # bin lick positions
        hist, bin_edges = np.histogram(np.digitize(lick_pos, np.arange(start=-10, stop=111, step=bin_size)),
                                       bins=np.arange(start=1, stop=int(120/bin_size+2)), density=normalized)
    return hist


def extract_performance_from_merged(data, novel, buffer=2, valid=False, bin_size=1, use_reward=False):
    """
    Extracts behavior data from one merged_behavior.txt file (acquired through behavior_import.py).
    :param data: pd.DataFrame of the merged_behavior*.txt file
    :param novel: bool, flag whether file was performed in novel corridor (changes reward zone location)
    :param buffer: int, position bins around the RZ that are still counted as RZ for licking
    :param valid: bool, flag whether trial was a RZ position validation trial (training corridor with shifted RZs)
    :param bin_size: int, bin size in VR units for binned licking performance analysis (divisible by zone borders)
    :param use_reward: bool flag whether to use valve openings to calculate number of passed reward zones. More
                        sensitive for well performing mice, but vulnerable against manual valve openings and
                        useless for autoreward trials.
    :returns lick_ratio: float, ratio between individual licking bouts that occurred in reward zones div. by all licks
    :returns stop_ratio: float, ratio between stops in reward zones divided by total number of stops
    """
    # set borders of reward zones
    if novel:
        zone_borders = np.array([[9, 19], [34, 44], [59, 69], [84, 94]])
    elif valid:
        zone_borders = np.array([[-6, 4], [34, 44], [66, 76], [90, 100]])
    else:
        zone_borders = np.array([[-6, 4], [26, 36], [58, 68], [90, 100]])

    zone_borders[:, 0] -= buffer
    zone_borders[:, 1] += buffer

    # Find out which reward zones were passed (reward given) if reward data is available
    reward_from_merged = False
    if use_reward and any(data['reward'] == -1):
        rz_passed = np.zeros(len(zone_borders))
        for idx, zone in enumerate(zone_borders):
            rz_data = data.loc[(data['VR pos'] > zone[0]) & (data['VR pos'] < zone[1]), 'reward']
            # Cap reward at 1 per reward zone (ignore possible manual water rewards given)
            if rz_data.sum() >= 1:
                rz_passed[idx] = 1
            else:
                rz_passed[idx] = 0
        passed_rz = rz_passed.sum()/len(zone_borders)
        reward_from_merged = True

    # Get indices of proper columns and transform DataFrame to numpy array for easier processing
    lick_idx = data.columns.get_loc('licks')
    enc_idx = data.columns.get_loc('encoder')
    pos_idx = data.columns.get_loc('VR pos')
    data = np.array(data)

    ### GET LICKING DATA ###
    # select only time point where the mouse licked
    lick_only = data[np.where(data[:, lick_idx] == 1)]

    if lick_only.shape[0] == 0:
        lick_ratio = np.nan  # set nan, if there was no licking during the trial
        if not reward_from_merged:
            passed_rz = 0
    else:
        # remove continuous licks that were longer than 5 seconds
        diff = np.round(np.diff(lick_only[:, 0]) * 10000).astype(int)  # get an array of time differences
        licks = np.split(lick_only, np.where(diff > 5)[0] + 1)  # split at points where difference > 0.5 ms (sample gap)
        licks = [i for i in licks if i.shape[0] <= 10000]      # only keep licks shorter than 5 seconds (10,000 samples)
        if len(licks) > 0:
            licks = np.vstack(licks)    # put list of arrays together to one array
            # out of these, select only time points where the mouse was in a reward zone
            lick_zone_only = []
            for zone in zone_borders:
                lick_zone_only.append(licks[(zone[0] <= licks[:, pos_idx]) & (licks[:, pos_idx] <= zone[1])])
            zone_licks = np.vstack(lick_zone_only)
            # the length of the zone-only licks divided by the all-licks is the zone-lick ratio
            lick_ratio = zone_licks.shape[0]/lick_only.shape[0]

            # correct by fraction of reward zones where the mouse actually licked
            if not reward_from_merged:
                passed_rz = len([x for x in lick_zone_only if len(x) > 0])/len(zone_borders)
            lick_ratio = lick_ratio * passed_rz

            # # correct by the fraction of time the mouse spent in reward zones vs outside
            # rz_idx = 0
            # for zone in zone_borders:
            #     rz_idx += len(np.where((zone[0] <= data[:, 1]) & (data[:, 1] <= zone[1]))[0])
            # rz_occupancy = rz_idx/len(data)
            # lick_ratio = lick_ratio/rz_occupancy

        else:
            lick_ratio = np.nan
            if not reward_from_merged:
                passed_rz = 0

    ### GET STOPPING DATA ###
    # select only time points where the mouse was not running (encoder between -2 and 2)
    stop_only = data[(-2 <= data[:, enc_idx]) & (data[:, enc_idx] <= 2)]
    # split into discrete stops
    diff = np.round(np.diff(stop_only[:, 0]) * 10000).astype(int) # get an array of time differences
    stops = np.split(stop_only, np.where(diff > 5)[0] + 1)  # split at points where difference > 0.5 ms (sample gap)
    # select only stops that were longer than 100 ms (200 samples)
    stops = [i for i in stops if i.shape[0] >= 200]
    # select only stops that were inside a reward zone (min or max position was inside a zone border)
    zone_stop_only = []
    for zone in zone_borders:
        zone_stop_only.append([i for i in stops if zone[0] <= np.max(i[:, pos_idx]) <= zone[1] or
                               zone[0] <= np.min(i[:, pos_idx]) <= zone[1]])
    # the number of the zone-only stops divided by the number of the total stops is the zone-stop ratio
    zone_stops = np.sum([len(i) for i in zone_stop_only])
    stop_ratio = zone_stops/len(stops)

    # Licks per bin
    licked_rz_bins = 0
    licked_nonrz_bins = 0
    bins = np.arange(start=-10, stop=111, step=bin_size)   # create bin borders for position bins (2 steps/6cm per bin)
    zone_bins = []
    for zone in zone_borders:
        zone_bins.extend(np.arange(start=zone[0], stop=zone[1]+1, step=bin_size))
    bin_idx = np.digitize(data[:, pos_idx], bins)
    for curr_bin in np.unique(bin_idx):
        if sum(data[np.where(bin_idx == curr_bin)[0], 2]) >= 1:
            if bins[curr_bin-1] in zone_bins:
                licked_rz_bins += 1
            else:
                licked_nonrz_bins += 1
    try:
        binned_lick_ratio = (licked_rz_bins/(licked_rz_bins+licked_nonrz_bins)) * passed_rz
    except ZeroDivisionError:
        binned_lick_ratio = 0

    return lick_ratio, binned_lick_ratio, stop_ratio


def collect_log_performance_data(root, novel, norm_date='0'):
    """
    Gathers and combines performance data from log files. Results are returned as one entry per session in three lists that can be
    used for plotting: Dates of sessions, performance speed and total and hit zone count.
    Works for folder and nonfolder structures. Called by single_mouse_performance once per mouse.
    :param root: base directory which holds all session folders (usually called M15, M16 etc.)
    :param novel: bool flag whether behavior in the novel or training context should be analysed
    :param norm_date: str; which date should the time line be normalized to (day 0, all other days calculated
    accordingly). If None, no normalization is performed and dates are returned as folder names. If '0', dates
    are normalized to the first Log file found. If datestr in the format 'YYYYMMDD', dates are normalized to this date.
    :return dates, speed, zones: lists that hold results in one entry per session. Can be indexed equivalently.
    """

    novel_vr = ['20191122a', '20191122b', '20191125', '20191126b', '20191127a', '20191128b', '20191129a',
                '20191203 prestroke', '20191203 poststroke', '20191204', '20191205', '20191206', '20191207',
                '20191208', '20191209', '20191210', '20191213', '20191216']
    #todo: change hard coding of novel vs training sessions (read from TDT files?)

    session_list = os.listdir(root)
    dates = list(np.zeros(len(session_list), dtype=int))
    speed = list(np.zeros(len(session_list), dtype=int))
    zones = list(np.zeros(len(session_list), dtype=int))

    for i in range(len(session_list)):
        if session_list[i] in novel_vr and novel or session_list[i] not in novel_vr and not novel:
            path = os.path.join(root, session_list[i])
            for (dirpath, dirnames, filenames) in os.walk(path):
                ### COLLECT TRIAL DURATION TIMES AND REWARD ZONE PERFORMANCE ###
                if len(filenames) > 0:
                    log_name = [s for s in filenames if 'TDT LOG' in s]
                    if len(log_name) == 1:
                        log_path = os.path.join(dirpath, log_name[0])
                        zone, all_zones, durations = extract_zone_performance_from_log_file(log_path)
                        speed[i] = durations
                        zones[i] = (zone, all_zones)
                    else:               # if no log file is present, try to find trial_duration files
                        dur_files = []
                        for step in os.walk(dirpath):   # go through all subfolders and look for trial_duration files
                            dur_file = [s for s in step[2] if 'trial_duration' in s]
                            if len(dur_file) == 1 and os.path.join(step[0], dur_file[0]) not in dur_files:
                                # if a new file is found, append it to the list
                                dur_files.append(os.path.join(step[0], dur_file[0]))
                        zones[i] = 0
                        if len(dur_files) > 0:
                            speed[i] = get_trial_duration(dur_files)
                        else:
                            print(f'No Log or trial_duration file found in {dirpath}!')
                            speed[i] = 0
                dates[i] = session_list[i]
                break  # only do one step with os.walk

    # only keep sessions with a date (only sessions with the correct corridor)
    mask = [True for i in range(len(dates))]
    for i in range(len(dates)):
        if type(dates[i]) == np.int32:
            mask[i] = False
    speed = list(compress(speed, mask))
    dates = list(compress(dates, mask))
    zones = list(compress(zones, mask))

    # normalize session dates
    if norm_date is not None:
        norm_days = normalize_dates(dates, norm_date)
    else:
        norm_days = dates

    return norm_days, speed, zones


def extract_zone_performance_from_log_file(path):
    """
    Reads a LabView TDT LOG file line by line and extracts reward zone performance by counting the number of individual
    zone encounters and how many zones have been passed (passed if the valve opened inside a zone). Results are returned
    in the zone array: one row per zone, columns: number of zone encounters (col 0) and number of passed zones (col 1).
    Called by collect_performance_data once for every session that has a log file.
    :param path: str, path of the log file
    :return zone, all_zones: arrays, zone stores data from individual zones, all_zones contains sum of zone columns
    :return duration: list of floats showing the duration of every trial
    """
    # zones saves the number each reward zone has been encountered (left column) and how many times
    # this reward zone has been hit (valve opened). One row per reward zone.
    zone = np.zeros((4, 2))
    temp_zones = np.zeros((4, 2))  # temp counter for each trial prevents counting of incomplete trials
    duration = []   # this list saves the duration of every trial as floats to estimate running speed

    # initialize memory variables
    in_rew_zone = False  # flag that remembers whether the current line is inside a reward zone
    curr_rew_zone = None    # number of the current reward zone
    already_opened = False  # flag that remembers if the valve already opened in the current reward zone
    start_time = None   # start time of the trial

    time_format = '%H:%M:%S.%f'

    with open(path, 'r') as log:
        lines = log.readlines()
        # Go through the log file line by line and register if the valve opened during a reward zone
        # reward zones are marked by the lines "With Cue X" and "VR leave Reward Zone:Y"
        for curr_line in lines:
            line = curr_line.split('\t')
            # when we enter a new trial, update zone counter, reset temporary counter and save start time
            if 'VR Task Begin' in line[-1]:
                zone += temp_zones
                temp_zones = np.zeros((4, 2))
                curr_time = dt.strptime(line[1], time_format)   # what is the time of the trial start?
                if start_time is not None:                      # do we have a start time of the previous trial?
                    duration.append((curr_time - start_time).total_seconds()) # what was the duration of this trial
                start_time = curr_time                          # remember the current time as the start time
            # we enter a reward zone...
            elif 'With Cue' in line[-1]:
                in_rew_zone = True  # set flag to True
                curr_rew_zone = int(line[-1][-2]) - 1  # which reward zone are we in?
                temp_zones[curr_rew_zone, 0] += 1  # add 1 to the number of zone encounters
            # a valve has been opened...
            elif 'Dev1/port0/line0-B' in line[-1]:
                if in_rew_zone and not already_opened:
                    temp_zones[curr_rew_zone, 1] += 1
                    already_opened = True
            # we leave a reward zone...
            elif 'VR leave Reward Zone' in line[-1]:
                in_rew_zone = False  # return flag to False
                curr_rew_zone = None  # return reward zone number to None
                already_opened = False  # reset flag

    all_zones = (np.sum(zone[:, 0]), np.sum(zone[:, 1]))    # tuple that includes sums of all passed reward zones

    return zone, all_zones, np.asarray(duration)


def get_trial_duration(file_list):
    """
    Extracts trial durations from trial_duration files (created during behavioral alignment).
    :param file_list: list of file paths of trial_duration.txt files, usually all files in one session
    :return dur: np.array of duration times of all trials in this session
    """
    dur = []
    for file in file_list:
        with open(file, 'r') as f:
            lines = f.readlines()
            for curr_line in lines:
                line = curr_line.split(':')
                try:
                    dur.append(int(line[-1]))
                except ValueError:
                    pass
    return np.array(dur)


def normalize_dates(date_list, norm_date):
    """
    Normalizes a list of dates by a given day_0 normalization date.
    :param date_list: list, contains strings of dates in format 'YYYYMMDD(X)',
                            X being a possible 'a' or 'b' for 2 sessions per day. If b, date is counted as 0.5 d later
    :param norm_date: str, shows the date to with all dates should be normalized to. If '0', dates are normalized to the
                            first date found. Otherwise, norm_date should be included in date_list.
    :return norm_days: list of numbers signalling the difference in days to the norm_date
    """

    day_0_idx = None
    # first, transform folder names to datetime objects that can be calculated with
    date_format = '%Y%m%d'
    days = []
    for date in date_list:
        if type(date) == str:
            if norm_date != '0' and date[:8] == norm_date:
                day_0_idx = date_list.index(date)  # if this day is the desired day0, remember its index
            curr_day = dt.strptime(date[:8], date_format)
            if date[-1] == 'b':
                curr_day = curr_day + delta(hours=12)  # add half a day for two-session days
            days.append(curr_day)
        else:
            days.append(np.nan)

    # if norm_date is 0, dates are normalized to the first date
    if norm_date == '0':
        day_0_idx = 0

    # if day_0_idx is still None, a normalization date could not be found and the raw list is returned
    if day_0_idx is None:
        print(f'Could not find normalization date of {norm_date}! No normalization possible.')
        return date_list
    # normalize days depending on the desired day 0
    else:
        day_0 = days[day_0_idx]  # find the date of the day_0
        norm_days = [(day - day_0) / delta(days=1) for day in days]  # set dates as differences from day_0 date
        for i in range(len(norm_days)):
            if norm_days[i] % 1 == 0:  # make whole days int to save space
                norm_days[i] = int(norm_days[i])

    if len(date_list) != len(norm_days):    # small sanity check
        return print('Normalized date list is not the same length as initial date list. Something went wrong!')
    else:
        return norm_days


def normalize_performance(data, session_range):
    """
    Normalize performance by dividing licking performance by the mean of the provided session window.
    :param data: DataFrame holding behavioral data, from load_performance_data()
    :param session_range: tuple of first and last session date (str) of the baseline window
    :return: input DataFrame with additional 'licking_binned_norm' column
    """
    session_idx = (int(data.loc[data['session_date'] == session_range[0], 'sess_id'].mean()),
                   int(data.loc[data['session_date'] == session_range[1], 'sess_id'].mean()))

    df_list = []
    for mouse in np.unique(data['mouse']):
        curr_df = data.loc[data['mouse'] == mouse]
        dat = curr_df.loc[(curr_df['sess_id'] >= session_idx[0]) & (curr_df['sess_id'] <= session_idx[1])]
        base = dat['licking_binned'].mean()
        curr_df['licking_binned_norm'] = curr_df['licking_binned'] / base
        df_list.append(curr_df)
    return pd.concat(df_list)

#%% Plotting

# todo make vertical line or something to signify stroke sessions


def quick_screen_session(path, valid=False):
    """
    Plots the binned running speed and licking of each trial in a session for a quick screening.
    TODO: fix vlines of RZs when mouse is running backwards (e.g. M57 20201126)
    :param path:
    :return:
    """

    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split('(\d+)', text)]

    if is_session_novel(path):
        zone_borders = np.array([[9, 19], [34, 44], [59, 69], [84, 94]])
    else:
        zone_borders = np.array([[-6, 4], [26, 36], [58, 68], [90, 100]])

    # go through all folders and subfolders and find merged_behavior files
    file_list = []
    for step in os.walk(path):
        if len(step[2]) > 0:
            for file in step[2]:
                if 'merged_behavior' in file and os.path.join(step[0], file) not in file_list:
                    file_list.append(os.path.join(step[0], file))
    file_list.sort(key=natural_keys)       # order files according to trial number (otherwise 11 is before 2)

    data_list = []
    for file in file_list:
        data_list.append(np.loadtxt(file))
    if len(data_list) == 0:
        return print(f'No trials found at {path}.')
    else:
        try:
            perf_old = int(10000 * np.mean(np.nan_to_num(np.loadtxt(os.path.join(path, 'performance.txt')))[:, 0]))/100
            perf_new = int(10000 * np.mean(np.nan_to_num(np.loadtxt(os.path.join(path, 'performance.txt')))[:, 1]))/100
        except OSError:
            perf_old = 'NaN'
            perf_new = 'NaN'
        except IndexError:
            perf_old = int(10000 * np.mean(np.nan_to_num(np.loadtxt(os.path.join(path, 'performance.txt')))[0]))/100
            perf_new = int(10000 * np.mean(np.nan_to_num(np.loadtxt(os.path.join(path, 'performance.txt')))[1]))/100
        # plotting
        bad_trials = []
        nrows = ceil(len(data_list)/3)
        ncols = 3
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 8))
        count = 0
        for row in range(nrows):
            for col in range(ncols):
                if count < len(data_list):
                    curr_trial = data_list[count]

                    # plot behavior
                    color = 'tab:red'
                    if nrows == 1:
                        ax[col].plot(-curr_trial[:, 4], color=color)       # plot running
                        ax[col].spines['top'].set_visible(False)
                        ax[col].spines['right'].set_visible(False)
                        ax[col].set_xticks([])
                        ax2 = ax[col].twinx()       # instantiate a second axes that shares the same x-axis
                    else:
                        ax[row, col].plot(-curr_trial[:, 4], color=color)  # plot running
                        ax[row, col].spines['top'].set_visible(False)
                        ax[row, col].spines['right'].set_visible(False)
                        ax[row, col].set_xticks([])
                        ax2 = ax[row, col].twinx()  # instantiate a second axes that shares the same x-axis
                    ax2.set_picker(True)
                    color = 'tab:blue'
                    ax2.plot(curr_trial[:, 2], color=color)       # plot licking
                    ax2.set_ylim(-0.1, 1.1)
                    ax2.set_ylabel(file_list[count])
                    ax2.axis('off')

                    # find samples inside reward zones
                    zones_idx = []
                    if valid and count > 4:     # New zone borders for validation trials (RZ shifted)
                        zone_borders = np.array([[-6, 4], [34, 44], [66, 76], [90, 100]])
                    for zone in zone_borders:
                        zones_idx.append(np.where((curr_trial[:, 1] > zone[0]) & (curr_trial[:, 1] < zone[1]))[0])
                    # show reward zone location
                    for zone in zones_idx:
                        ax2.axvspan(min(zone), max(zone), color='grey', alpha=0.2)
                    # mark valve openings
                    for valve in np.where(curr_trial[:,6]==1.0)[0]:
                        ax2.axvline(valve, c='yellow', linewidth=2)
                    count += 1

    def onpick(event):
        this_plot = event.artist  # save artist (axis) where the pick was triggered
        trial = this_plot.get_ylabel()
        if trial not in bad_trials:
            bad_trials.append(trial)

    def closed(event):
        sort = sorted(bad_trials)
        print(*sort, sep='\n')

    fig.canvas.mpl_connect('close_event', closed)
    fig.canvas.mpl_connect('pick_event', onpick)
    plt.tight_layout()
    fig.suptitle(f'Performance: {perf_old}% (old), {perf_new}% (new)', fontsize=14)
    plt.show()


def plot_single_mouse(input, mouse, field='licking', rotate_labels=False, session_range=None, scale=1, ax=None):
    """
    Plots the performance in % licks in RZ per session of one mouse.
    :param input: pandas DataFrame from load_performance_data()
    :param mouse: str, ID of the mouse whose performance should be plotted (according to input['mouse'])
    :param rotate_labels: bool flag whether x-axis labels should be rotated by 45°
    :param session_range: optional tuple or list, restricted range of sessions to be displayed (from input['sess_id'])
    :return:
    """
    sns.set()
    sns.set_style('whitegrid')
    data = deepcopy(input)
    data[field] = data[field] * 100

    if ax is None:
        plt.figure()
    else:
        plt.sca(ax)

    sessions = np.sort(data['sess_norm'].unique())
    ax = sns.lineplot(x='sess_id', y=field, data=data[data['mouse'] == mouse], label=mouse)
    if session_range is None:
        ax.set(ylim=(0, 100), ylabel='licks in reward zone [%]', xlabel='day', xticks=range(len(sessions)),
               xticklabels=sessions)
    else:
        ax.set(ylim=(0, 100), xlim=(session_range[0], session_range[1]), ylabel='licks in reward zone [%]',
               xlabel='day', xticks=range(len(sessions)), xticklabels=sessions)
    if rotate_labels:
        labels = ax.get_xticklabels()
        ax.set_xticklabels(labels, rotation=45, ha='right')

    sns.set(font_scale=scale)

    return ax


def plot_all_mice_avg(input, field='licking', rotate_labels=False, session_range=None, scale=1):
    """
    Plots the performance in % licks in RZ per session averaged over all mice.
    :param input: pandas DataFrame from load_performance_data()
    :param rotate_labels: bool flag whether x-axis labels should be rotated by 45°
    :param session_range: optional tuple or list, restricted range of sessions to be displayed (from input['sess_id'])
    :return:
    """
    sns.set()
    sns.set_style('whitegrid')
    data = deepcopy(input)
    # data[field] = data[field] * 100
    plt.figure()
    sessions = np.sort(data['sess_norm'].unique())
    ax = sns.lineplot(x='sess_id', y=field, data=data)

    if session_range is None:
        ax.set(ylabel='licks in reward zone [%]',
               title='Average of all mice', xticks=range(len(sessions)), xticklabels=sessions)
    else:
        ax.set(xlim=(session_range[0], session_range[1]), ylabel='licks in reward zone [%]',
               title='Average of all mice', xticks=range(len(sessions)), xticklabels=sessions)

    if rotate_labels:
        labels = ax.get_xticklabels()
        ax.set_xticklabels(labels, rotation=45, ha='right')
    sns.set(font_scale=scale)


def plot_all_mice_separately(input, field='licking_binned', x_axis='sess_norm', frac=None, rotate_labels=False,
                             session_range=None, scale=1, hlines=None, vlines=None, columns=3):
    """
    Plots the performance in % licks in RZ per session of all mice in separate graphs.
    :param input: pandas DataFrame from load_performance_data()
    :param field: column of the input DataFrame that should be plotted on the y-axis
    :param x_axis: column of the input DataFrame that should be plotted on the x-axis
    :param fract: int, shows first and last 1/fract of trials separately per mouse
    :param rotate_labels: bool flag whether x-axis labels should be rotated by 45°
    :param session_range: optional tuple or list, restricted range of sessions to be displayed (from input['sess_id'])
    :param scale: int, scaling factor of axis and tick labels applied by seaborn
    :param hlines: optional tuple or list, y-axis positions for green horizontal lines (e.g. pre-stroke baselines)
    :param vlines: optional tuple or list, x-axis positions for red vertical lines (e.g. stroke dates)
    :param columns: int, number of columns of the figure
    :return:
    """

    data = deepcopy(input)
    data[field] = data[field] * 100
    data["licking"] = data["licking"] * 100
    # Give each trial a label, whether its in the first or last fraction of the session
    if frac is not None:
        data["fraction"]="all"
        for mouse in data.mouse.unique():
            for sess in data.session_date.unique():
                # Get indices of the current session
                curr_sess = data[(data.mouse == mouse) & (data.session_date == sess)]
                # Get indices of first and last fraction of the session
                quant = int(np.ceil(len(curr_sess.index)/frac))
                if quant < 2:
                    quant = 2
                # set label of corresponding rows
                data.iloc[curr_sess.index[:quant], data.columns.get_loc("fraction")] = "first"
                data.iloc[curr_sess.index[-quant:], data.columns.get_loc("fraction")] = "last"
        # duplicate rows with "first" and "last", set "fraction" to "all" so that it is displayed in the "all" hue
        rows = data.loc[data.fraction != "all"]
        rows["fraction"] = "all"
        data = pd.concat([data, rows])


    sns.set()
    sns.set_style('whitegrid')
    sessions = np.array(np.sort(data[x_axis].unique()), dtype=int)
    if len(data['mouse'].unique()) < columns:
        columns = len(data['mouse'].unique())
    if frac is not None:
        grid = sns.FacetGrid(data, col='mouse', hue="fraction", col_wrap=columns, height=3, aspect=2, legend_out=False)
        grid.map(sns.lineplot, 'sess_id', field).add_legend()
    else:
        grid = sns.FacetGrid(data, col='mouse', col_wrap=columns, height=3, aspect=2)

    # grid.map(sns.lineplot, 'sess_id', field, hue="fraction")
    grid.set_axis_labels('session', 'licks in reward zone [%]')

    # Plot red vertical lines (signalling stroke) if provided
    if vlines is not None:
        x = np.sort(input[x_axis].unique()).astype(float)
        for line in vlines:
            # Find actual x-axis value of the vline based on current x-axis labels
            curr_x = int(line)
            adj_x = np.where(x == curr_x)[0]
            if len(adj_x) == 1:
                for ax in grid.axes.ravel():
                    # Draw line and offset actual x-value by x-label
                    ax.axvline(adj_x[0]+(line-curr_x), color='r')
            else:
                raise ValueError(f'Provided vline position {line} is not a label on the X-axis!')

    # Plot green horizontal line (signaling baseline) if provided (one per mouse/axes)
    if hlines is not None:
        hlines = [x*100 for x in hlines]
        for idx, ax in enumerate(grid.axes.ravel()):
            ax.axhline(hlines[idx], color='g')

    if session_range is None:
        out = grid.set(ylim=(0, 100), xticks=np.sort(input['sess_id'].unique()), xticklabels=sessions)
    else:
        out = grid.set(ylim=(0, 100), xlim=(session_range[0], session_range[1]),
                       xticks=range(len(sessions)), xticklabels=sessions)
    for ax in grid.axes.ravel():
        if rotate_labels:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    sns.set(font_scale=scale)
    plt.tight_layout()


def plot_validation_performance(path, change_trial=5, bin_size=1, normalized=True, cmap='jet', validation_zones=False,
                                show_old=False, target=None):
    """
    Plots the performance, individual licks per bin and binary licks per bin of one validation session.
    :param path: str, directory of one session
    :param change_trial: int, number of trial when the condition changed (default 5)
    :param bin_size: int, size of VR position bins (default 1 for standard VR position from 0 to 120)
    :param normalized: bool flag whether performance should be normalized to the mean pre-change performance
    :param cmap: str, colormap for the individual-licks-plot
    :param validation_zones: bool flag whether to plot new validation zones (for sessions when zone positions shifted)
    :param show_old: bool flag whether to plot the old method of performance calculation (no binned licking)
    :param target: str, if provided plots are saved to this directory and closed afterwards
    :return:
    """
    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split('(\d+)', text)]

    zone_borders_new = np.array([[-6, 4], [34, 44], [66, 76], [90, 100]])+10
    zone_borders_old = np.array([[-6, 4], [26, 36], [58, 68], [90, 100]])+10

    mouse = path.split(sep=os.path.sep)[-1]

    file_list = glob(path+'\\*\\merged_behavior*.txt')
    if len(file_list) == 0:
        file_list = glob(path + '\\merged_behavior*.txt')
    file_list.sort(key=natural_keys)

    data = np.zeros((len(file_list), int(120/bin_size)))
    for idx, file in enumerate(file_list):
        data[idx] = get_binned_licking(np.loadtxt(file), bin_size=bin_size, normalized=normalized)

    #### Plot individually counted binned licks
    fig = plt.figure(figsize=(12, 7))
    out = plt.pcolormesh(data, cmap=cmap)

    # shade areas of old and new reward zone locations
    for old, new in zip(zone_borders_old, zone_borders_new):
        out.axes.axvspan(min(old), max(old), color='magenta', alpha=0.2)
        if validation_zones:
            out.axes.axvspan(min(new), max(new), color='yellow', alpha=0.2)

    # design fixes
    out.axes.axhline(change_trial, color='r')
    out.axes.invert_yaxis()
    out.axes.tick_params(labelsize=12)
    # out.axes.set_title(path[57:60]+', '+path[61:], fontsize=18)
    out.axes.set_ylabel('Trial', rotation=90, fontsize=15)
    out.axes.set_xlabel('VR position [a.u.]', fontsize=15)
    fig.tight_layout()

    if target is not None:
        plt.savefig(os.path.join(target, f'{mouse}_binned_licking.png'))
        plt.close()

    #### Plot binary binned licks
    data[data > 0] = 1
    fig = plt.figure(figsize=(12, 7))
    out = plt.pcolormesh(data, cmap='Blues')

    # shade areas of old and new reward zone locations
    for old, new in zip(zone_borders_old, zone_borders_new):
        out.axes.axvspan(min(old), max(old), color='blue', alpha=0.2)
        if validation_zones:
            out.axes.axvspan(min(new), max(new), color='red', alpha=0.2)

    # design fixes
    out.axes.axhline(change_trial, color='r')
    out.axes.invert_yaxis()
    out.axes.tick_params(labelsize=12)
    # out.axes.set_title(path[57:60]+', '+path[61:], fontsize=18)
    out.axes.set_ylabel('Trial', rotation=90, fontsize=15)
    out.axes.set_xlabel('VR position [a.u.]', fontsize=15)
    fig.tight_layout()

    if target is not None:
        plt.savefig(os.path.join(target, f'{mouse}_binned_binary_licking.png'))
        plt.close()

    #### Plot performance
    perf_data = np.nan_to_num(np.loadtxt(os.path.join(path, 'performance.txt')))
    fig = plt.figure(figsize=(12, 4))
    if show_old:
        out = plt.plot(perf_data[:, 0])
        out[0].set_label('old')
    out2 = plt.plot(perf_data[:, 1])
    out2[0].set_label('new')
    plt.ylim(0, 1.1)
    out2[0].axes.tick_params(labelsize=10)
    out2[0].axes.set_ylabel('Licking performance', rotation=90, fontsize=12)
    out2[0].axes.set_xlabel('Trial', fontsize=12)
    out2[0].axes.axvline(change_trial-0.5, color='r')
    if show_old:
        out2[0].axes.legend()
    fig.tight_layout()
    if target is not None:
        plt.savefig(os.path.join(target, f'{mouse}_performance.png'))
        plt.close()


def plot_binned_licking(path, bin_size=2, novel=False):
    """
    Plots the licking of a single session as a histogram binned over VR positions.
    :param path: str, path of session
    :param bin_size: int, amount of binning
    :param novel: bool flag whether session was in the novel corridor
    :return:
    """

    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split('(\d+)', text)]

    if novel:
        zone_borders = np.array([[9, 19], [34, 44], [59, 69], [84, 94]])
    else:
        zone_borders = np.array([[-6, 4], [26, 36], [58, 68], [90, 100]]) + 10

    mouse = path.split(sep=os.path.sep)[-2]
    session = path.split(sep=os.path.sep)[-1]

    file_list = glob(path + '\\*\\merged_behavior*.txt')
    if len(file_list) == 0:
        file_list = glob(path + '\\merged_behavior*.txt')
    file_list.sort(key=natural_keys)

    data = np.zeros((len(file_list), int(120 / bin_size)))
    for idx, file in enumerate(file_list):
        data[idx] = get_binned_licking(np.loadtxt(file), bin_size=bin_size, normalized=False)

    data[data > 0] = 1

    plt.figure(figsize=(8, 4))
    plt.hist(np.linspace(0, 400, 60), bins=60, weights=(np.sum(data, axis=0) / len(data)) * 100,
             facecolor='black', edgecolor='black')

    for zone in zone_borders:
        plt.axvspan(zone[0] * (10 / 3), zone[1] * (10 / 3), color='red', alpha=0.3)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(0, 105)
    plt.xlabel('VR position', fontsize=22)
    plt.ylabel('Licked in bin [%]', fontsize=22)
    plt.tight_layout()
    plt.title(f'Mouse {mouse}, session {session}', fontsize=24)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
