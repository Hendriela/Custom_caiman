import os
import numpy as np
import seaborn as sns
from datetime import datetime as dt
from datetime import timedelta as delta
from itertools import compress
import matplotlib.pyplot as plt


def multi_mouse_performance(mouse_dir_list, precise_duration=True, separate_zones=False, date_0=None):
    if date_0 is None or date_0 == '0':
        for mouse in mouse_dir_list:
            single_mouse_performance(mouse, precise_duration, separate_zones, date_0)
    elif type(date_0) == list:
        if len(mouse_dir_list) == len(date_0):
            for i in range(len(mouse_dir_list)):
                single_mouse_performance(mouse_dir_list[i], precise_duration, separate_zones, date_0[i])
        else:
            print('If a list, date_0 has to be the same length as mouse_dir_list!')
    else:
        print('Date_0 input not understood.')


def single_mouse_performance(root, precise_duration=False, separate_zones=False, date_0=None):
    # get performance data
    dates, speed, zones = collect_performance_data(root, norm_dates=date_0)

    # plot trial duration as a line plot (simple) or box plot (precise)
    plt.figure(figsize=(15, 8))
    plt.title(f'{root[-3:]}: Trial duration across sessions')
    if precise_duration:
        ax = sns.boxplot(data=speed)
    else:
        ax = sns.pointplot(data=speed)
    set_xlabels(ax, dates, date_0)
    out = ax.set_ylabel('Trial duration [s]')

    # plot hit_zone_ratio as line plot
    plt.figure()
    plt.title(f'{root[-3:]}: Reward zone hits [%] across sessions')
    if separate_zones:
        pass
    else:
        zone_ratio = np.array([(i[1][1]/i[1][0])*100 for i in zones])   # calculate hit ratio of all zones combined
        ax_zones = sns.lineplot(data=zone_ratio, marker=True)
        set_xlabels(ax_zones, dates, date_0)

    #Todo: implement other performance parameters (licks, stops)

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


def collect_performance_data(root, norm_dates='0'):
    """
    Gathers and combines performance data from log files (reward zone hit rate) and 'trial_duration' (performance speed)
    files created during behavioral alignment. Results are returned as one entry per session in three lists that can be
    used for plotting: Dates of sessions, performance speed and total and hit zone count.
    Works for folder and nonfolder structures. Called by single_mouse_performance once per mouse.
    :param root: base directory which holds all session folders (usually called M15, M16 etc.)
    :param norm_dates: str; which date should the time line be normalized to (day 0, all other days calculated
    accordingly). If None, no normalization is performed and dates are returned as folder names. If '0', dates
    are normalized to the first Log file found. If datestr in the format 'YYYYMMDD', dates are normalized to this date.
    :return dates, speed, zones: lists that hold results in one entry per session. Can be indexed equivalently.
    """
    session_list = os.listdir(root)
    dates = list(np.zeros(len(session_list)))
    speed = list(np.zeros(len(session_list)))
    zones = list(np.zeros(len(session_list)))

    for i in range(len(session_list)):
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
                else:               # if no log file is present, return nan to later filter it out
                    zones[i] = 0
                    speed[i] = 0

            dates[i] = session_list[i]

            break # only do one step with os.walk

    # filter out sessions without data
    mask = [True for i in range(len(dates))]
    for i in range(len(speed)):
        if type(speed[i]) == int:
            mask[i] = False
    speed = list(compress(speed, mask))
    dates = list(compress(dates, mask))
    zones = list(compress(zones, mask))

    # normalize session dates
    if norm_dates is not None:
        day_0_idx = None
        #first, transform folder names to datetime objects that can be calculated with
        date_format = '%Y%m%d'
        days = []
        for date in dates:
            if norm_dates != '0' and date[:8] == norm_dates:
                day_0_idx = dates.index(date)   # if this day is the desired day0, remember its index
            curr_day = dt.strptime(date[:8], date_format)
            if date[-1] == 'b':
                curr_day = curr_day + delta(hours=12)   # add half a day for two-session days
            days.append(curr_day)

        #normalize days depending on the desired day 0
        if norm_dates == '0':
            day_0_idx = 0   # if norm_dates was 0, dates are normalized to the first date
        if day_0_idx is None:
            print('Could not find normalization date! No normalization possible.')
            norm_days = dates
        else:
            day_0 = days[day_0_idx]
            norm_days = [(day - day_0)/delta(days=1) for day in days]
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
