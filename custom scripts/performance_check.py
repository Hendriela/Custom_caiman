import os
import numpy as np
import seaborn as sns

def multi_mouse_performance(mouse_dir_list):

    for mouse in mouse_dir_list:
        pass #TODO finish

def single_mouse_performance(root, precise_duration=False, separate_zones=False):
    # get performance data
    dates, speed, zones = collect_performance_data(root)
    # plot trial duration as a line plot (simple) or violin plot (precise)
    if precise_duration:
        ax_dur = sns.violinplot(x=dates["session date"], y=speed["Trial duration [s]"])
    else:
        ax_dur = sns.lineplot(x=dates["session date"], y=speed["Trial duration [s]"])

    if separate_zones:
        pass
    else:
        zone_ratio = [(i[1][1]/i[1][0])*100 for i in zones]   # calculate hit ratio of all zones combined
        ax_zones = sns.lineplot(x=dates["session date"], y=zone_ratio["Reward zones hit [%]"],
                                marker=True)
        #TODO finish

def collect_performance_data(root):
    """
    Gathers and combines performance data from log files (reward zone hit rate) and 'trial_duration' (performance speed)
    files created during behavioral alignment. Results are returned as one entry per session in three lists that can be
    used for plotting: Dates of sessions, performance speed and total and hit zone count.
    Works for folder and nonfolder structures. Called by single_mouse_performance once per mouse.
    :param root: base directory which holds all session folders (usually called M15, M16 etc.)
    :return dates, speed, zones: lists that hold results in one entry per session. Can be indexed equivalently.
    """
    session_list = os.listdir(root)
    dates = list(np.zeros(len(session_list)))
    speed = list(np.zeros(len(session_list)))
    zones = list(np.zeros(len(session_list)))

    for i in range(len(session_list)):
        path = os.path.join(root, session_list[i])

        for (dirpath, dirnames, filenames) in os.walk(path):
            ### COLLECT TRIAL DURATION TIMES ###
            if 'trial_duration.txt' in filenames:  # this is the case in trials without imaging (no subfolders per session)
                curr_performance = np.loadtxt(os.path.join(path, 'trial_duration.txt'), skiprows=1) # load the trial times

            elif len(dirnames) > 0: # subfolders in one session occur during imaging trials
                # here, trial times have to be collected across different network folders and added to one array
                for j in range(len(dirnames)):
                    curr_file_path = os.path.join(path, dirnames[j], 'trial_duration.txt')
                    if os.path.isfile(curr_file_path):
                        if j == 0:
                            curr_performance = np.loadtxt(curr_file_path, skiprows=1)  # load the trial times
                        else:
                            curr_performance = np.hstack((curr_performance, np.loadtxt(curr_file_path, skiprows=1)))
                    else:
                        curr_performance = 0
            else:
                print(f'No performance data available at {path}')
                break

            dates[i] = session_list[i]
            speed[i] = curr_performance

            ### COLLECT REWARD ZONE HIT PERCENTAGE ###
            log_file = [i for i in filenames if 'TDT LOG' in i]

            if len(log_file) > 0:
                zone =
                for file in log_file:
                    log_path = os.path.join(path, file)
                    zone, all_zones = extract_zone_performance_from_log_file(log_path)
                    zones[i] = (zone, all_zones)
            else:
                zones[i] = np.nan  # if no log file is present, return nan to later filter it out

            break # only do one step with os.walk
    return dates, speed, zones


def extract_zone_performance_from_log_file(path):
    """
    Reads a LabView TDT LOG file line by line and extracts reward zone performance by counting the number of individual
    zone encounters and how many zones have been passed (passed if the valve opened inside a zone). Results are returned
    in the zone array: one row per zone, columns: number of zone encounters (col 0) and number of passed zones (col 1).
    Called by collect_performance_data once for every session that has a log file.
    :param path: str, path of the log file
    :return zone, all_zones: arrays, zone stores data from individual zones, all_zones contains sum of zone columns
    """
    # zones saves the number each reward zone has been encountered (left column) and how many times
    # this reward zone has been hit (valve opened). One row per reward zone.
    zone = np.zeros((4, 2))
    in_rew_zone = False  # flag that remembers whether the current line is inside a reward zone
    curr_rew_zone = None
    temp_zones = np.zeros((4, 2))  # temp counter for each trial prevents counting of incomplete trials

    with open(path, 'r') as log:
        lines = log.readlines()
        # Go through the log file line by line and register if the valve opened during a reward zone
        # reward zones are marked by the lines "With Cue X" and "VR leave Reward Zone:Y"
        for curr_line in lines:
            line = curr_line.split('\t')
            # when we enter a new trial, update zone counter and reset temporary counter
            if 'VR Task Begin' in line[-1]:
                zone += temp_zones
                temp_zones = np.zeros((4, 2))
            # we enter a reward zone...
            elif 'With Cue' in line[-1]:
                in_rew_zone = True  # set flag to True
                curr_rew_zone = int(line[-1][-2]) - 1  # which reward zone are we in?
                temp_zones[curr_rew_zone, 0] += 1  # add 1 to the number of zone encounters
            # we leave a reward zone...
            elif 'VR leave Reward Zone' in line[-1]:
                in_rew_zone = False  # return flag to False
                curr_rew_zone = None  # return reward zone number to None
            # a valve has been opened...
            elif 'Dev1/port0/line0-B' in line[-1]:
                if in_rew_zone and not already_licked:
                    temp_zones[curr_rew_zone, 1] += 1
    all_zones = (np.sum(zone[:, 0]), np.sum(zone[:, 1]))

    return zone, all_zones