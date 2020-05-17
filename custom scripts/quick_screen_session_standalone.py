import os
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import re
from glob import glob
import tkinter as tk
from tkinter import filedialog


def main():
    root = set_root()
    quick_screen_session(root)


def set_root():
    root = tk.Tk()
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)
    root_dir = filedialog.askdirectory(title='Select session folder that you want to screen')
    root_dir = root_dir.replace('/', '\\')
    root.withdraw()
    print(f'Root directory:\n {root_dir}')
    return root_dir


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

        perf = np.nan_to_num(np.loadtxt(os.path.join(path, 'performance.txt')))
        # plotting
        bad_trials = []
        nrows = ceil(len(data_list)/3)
        ncols = 3
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 8))
        fig.suptitle('Performance: {:2.2f}%'.format(100*np.mean(perf[:, 0])), fontsize=14)
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
        print('\nYou selected the following trials...')
        print(*sort, sep='\n')

    fig.canvas.mpl_connect('close_event', closed)
    fig.canvas.mpl_connect('pick_event', onpick)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
