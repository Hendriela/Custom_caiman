import os
import numpy as np
import seaborn as sns
from datetime import datetime as dt
from datetime import timedelta as delta
from itertools import compress, chain
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from math import ceil


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


def load_performance_data(roots, novel, norm_date):
    """
    Collects licking and stopping data from merged_behavior.txt files for a list of mice.
    :param roots: str, path to the batch folder (which contains folders for each mouse)
    :param novel: bool, whether novel or training sessions should be analysed
    :param norm_date: date where session dates should be normalized (one entry per mouse or single entry for all mice).
                        'None' = no normalization (dates labelled with session folder name),
                        '0' = normalization to the first session,
                        str in format 'YYYYMMDD' = norm. to the session of the respective day (neg for previous days).
    """
    data = []
    for root in roots:
        for step in os.walk(root):
            # Find performance
            if 'performance.txt' in step[2] and novel == is_session_novel(step[0]):

                # load data of the current session and add it to the global list as a pd.Series
                file_path = os.path.join(step[0], 'performance.txt')
                sess_data = np.loadtxt(file_path)
                sess_data = np.nan_to_num(sess_data)
                if len(sess_data.shape) == 1:
                    lick = [sess_data[0]]
                    stop = [sess_data[1]]
                elif len(sess_data.shape) > 1:
                    lick = sess_data[:, 0]
                    stop = sess_data[:, 1]
                else:
                    raise ValueError(f'No data found in {file_path}.')

                sess = step[0].split(os.path.sep)[-1]
                mouse = step[0].split(os.path.sep)[-2]

                # store data of this session and this mouse in a temporary DataFrame which is added to the global list
                data.append(pd.DataFrame({'licking': lick, 'stopping': stop, 'session': sess, 'mouse': mouse}))

    df = pd.concat(data, ignore_index=True)

    # give sessions a continuous id for plotting
    all_sess = sorted(df['session'].unique())
    count = 0
    df['sess_id'] = -1
    for session in all_sess:
        df.loc[df['session'] == session, 'sess_id'] = count
        count += 1

    # normalize days if necessary # todo: fix with new DF and dict structure
    if type(norm_date) == dict:
        for key in norm_date:
            if norm_date[key] is not None:
                df_mask = df['mouse'] == key
                session_norm = normalize_dates(df['mouse'] == key, norm_date[key])
            else:
                session_norm = sess

    return df


def save_multi_performance(path, overwrite=False):
    """
    Wrapper function for save_performance_data that goes through folders and looks for sessions that have no
    performance.txt yet. Session folders are determined by the presence of the LOG.txt file.
    :param path: Top-level directory where subdirectories are searched.
    :param path: bool flag whether to overwrite existing performance.txt files.
    """
    for (dirpath, dirnames, filenames) in os.walk(path):
        if len([x for x in filenames if 'TDT LOG' in x]) == 1:
            if 'performance.txt' not in filenames or overwrite:
                print(f'Computing performance of session {dirpath}.')
                save_performance_data(dirpath)
    print('Everything processed!')


def save_performance_data(session):
    """
    Calculates and saves licking and stopping ratios of a session.
    :param session: str, path to the session folder that holds behavioral txt files
    :return:
    """
    # Get the list of all behavior files in the session folder
    file_list = list()
    for (dirpath, dirnames, filenames) in os.walk(session):
        file_list += glob(dirpath + '\\merged_behavior*.txt')

    session_performance = np.zeros((len(file_list), 2))
    for i in range(len(file_list)):
        session_performance[i, 0], session_performance[i, 1] = \
            extract_performance_from_merged(np.loadtxt(file_list[i]), novel=is_session_novel(session), buffer=2)

    file_path = os.path.join(session, f'performance.txt')
    np.savetxt(file_path, session_performance, delimiter='\t',  fmt=['%.4f', '%.4f'], header='Licking\tStopping')


def is_session_novel(path):
    # check whether the current session is in the novel corridor
    context = None
    if len(glob(path + '\\TDT LOG*')) == 1:
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


def quick_screen_session(path):
    """
    Plots the binned running speed and licking of each trial in a session for a quick screening.
    :param path:
    :return:
    """
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

    data_list = []
    for file in file_list:
        data_list.append(np.loadtxt(file))
    if len(data_list) == 0:
        return print(f'No trials found at {path}.')
    else:
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
                    count += 1

                    # find samples inside reward zones
                    zones_idx = []
                    for zone in zone_borders:
                        zones_idx.append(np.where((curr_trial[:, 1] > zone[0]) & (curr_trial[:, 1] < zone[1]))[0])
                    # show reward zone location
                    for zone in zones_idx:
                        ax2.axvspan(min(zone), max(zone), color='grey', alpha=0.2)

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
    plt.show()


def extract_performance_from_merged(data, novel, buffer):
    """
    Extracts behavior data from one merged_behavior.txt file (acquired through behavior_import.py).
    :param data: np.array of the merged_behavior*.txt file
    :param novel: bool, flag whether file was performed in novel corridor (changes reward zone location)
    :param buffer: int, position bins around the RZ that are still counted as RZ for licking
    :returns lick_ratio: float, ratio between individual licking bouts that occurred in reward zones div. by all licks
    :returns stop_ratio: float, ratio between stops in reward zones divided by total number of stops
    """
    # set borders of reward zones
    if novel:
        zone_borders = np.array([[9, 19], [34, 44], [59, 69], [84, 94]])
    else:
        zone_borders = np.array([[-6, 4], [26, 36], [58, 68], [90, 100]])

    zone_borders[:, 0] -= buffer
    zone_borders[:, 1] += buffer


    ### GET LICKING DATA ###
    # select only time point where the mouse licked
    lick_only = data[np.where(data[:, 2] == 1)]

    if lick_only.shape[0] == 0:
        lick_ratio = np.nan # set nan, if there was no licking during the trial
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
                lick_zone_only.append(licks[(zone[0] <= licks[:, 1]) & (licks[:, 1] <= zone[1])])
            zone_licks = np.vstack(lick_zone_only)
            # the length of the zone-only licks divided by the all-licks is the zone-lick ratio
            lick_ratio = zone_licks.shape[0]/lick_only.shape[0]

            # correct by fraction of reward zones where the mouse actually licked
            passed_rz = len([x for x in lick_zone_only if len(x) > 0])
            lick_ratio = lick_ratio * (passed_rz / len(zone_borders))

            # # correct by the fraction of time the mouse spent in reward zones vs outside
            # rz_idx = 0
            # for zone in zone_borders:
            #     rz_idx += len(np.where((zone[0] <= data[:, 1]) & (data[:, 1] <= zone[1]))[0])
            # rz_occupancy = rz_idx/len(data)
            # lick_ratio = lick_ratio/rz_occupancy

        else:
            lick_ratio = np.nan

    ### GET STOPPING DATA ###
    # select only time points where the mouse was not running (encoder between -2 and 2)
    stop_only = data[(-2 <= data[:, -1]) & (data[:, -1] <= 2)]
    # split into discrete stops
    diff = np.round(np.diff(stop_only[:, 0]) * 10000).astype(int) # get an array of time differences
    stops = np.split(stop_only, np.where(diff > 5)[0] + 1)  # split at points where difference > 0.5 ms (sample gap)
    # select only stops that were longer than 100 ms (200 samples)
    stops = [i for i in stops if i.shape[0] >= 200]
    # select only stops that were inside a reward zone (min or max position was inside a zone border)
    zone_stop_only = []
    for zone in zone_borders:
        zone_stop_only.append([i for i in stops if zone[0] <= np.max(i[:, 1]) <= zone[1] or zone[0] <= np.min(i[:, 1]) <= zone[1]])
    # the number of the zone-only stops divided by the number of the total stops is the zone-stop ratio
    zone_stops = np.sum([len(i) for i in zone_stop_only])
    stop_ratio = zone_stops/len(stops)

    return lick_ratio, stop_ratio


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

#%% Plotting


def plot_single_mouse(data, mouse):
    plt.figure()
    ax = sns.lineplot(x='session', y='licking', data=data[data['mouse'] == mouse])
    ax.set(ylim=(0, 1), ylabel='licks in reward zone [%]', title=mouse)


def plot_all_mice_avg(data, rotate_labels=False):
    plt.figure()
    ax = sns.lineplot(x='session', y='licking', data=data)
    ax.set(ylim=(0, 1), ylabel='licks in reward zone [%]', title='Average of all mice')
    if rotate_labels:
        labels = ax.get_xticklabels()
        ax.set_xticklabels(labels, rotation=45, ha='right')


def plot_all_mice_separately(data, rotate_labels=False):
    sessions = data['session'].unique()
    grid = sns.FacetGrid(data, col='mouse', col_wrap=3, height=3, aspect=2)
    grid.map(sns.lineplot, 'sess_id', 'licking')
    grid.set_axis_labels('session', 'licks in reward zone [%]')
    out = grid.set(ylim=(0, 1), xlim=(-0.2, len(sessions) - 0.8), xticks=range(len(sessions)), xticklabels=sessions)
    if rotate_labels:
        for ax in grid.axes.ravel():
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()

