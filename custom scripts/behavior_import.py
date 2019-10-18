import numpy as np
import tkinter as tk
from tkinter import filedialog
import glob
import os
from math import ceil, floor
import sys


def load_file(path):
    """
    Loads files from a directory, with error messages in case there is none or more than 1 file with that name.
    :param path: directory
    :return: loaded file as np.array
    """
    file_path = glob.glob(path)
    if len(file_path) < 1:
        raise Exception(f'No files found at {path}!')
    elif len(file_path) > 1:
        raise Exception(f'File name ambiguous, multiple files found at {path}')
    else:
        return np.loadtxt(file_path[0])


def progress(count, total, status=''):
    bar_len = 20
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


frame_list = [4837, 11174, 709, 3275, 2036, 6274, 7654, 1622, 5475]
# 14 N1 frame_list = [900, 5927, 2430, 1814, 2504, 4624, 6132, 2168, 1953, 2670, 3817]  #todo remove hardcoding
#%% get directory of all trial-folders
root = tk.Tk()
root.lift()
root.attributes('-topmost', True)
root.after_idle(root.attributes, '-topmost', False)
root_dir = filedialog.askdirectory()
root.withdraw()
folder_list = np.sort(np.array(os.listdir(root_dir), dtype='int'))

#%% loop through every folder and process the files
counter = 1
for folder in folder_list:
    if len(frame_list) != len(folder_list):
        print('Different number of trials from frame_list and folder_list!')
        break
    else:
        print(f'\nNow processing trial {counter} of {len(folder_list)}: Folder {folder}...')
        curr_path = os.path.join(root_dir, f'{folder}')
        # load the three files (encoder (running speed), TCP (VR position) and TDT (licking + frame trigger))
        encoder = load_file(curr_path+r'\Encoder*.txt')
        position = load_file(curr_path+r'\TCP*.txt')
        trigger = load_file(curr_path+r'\TDT*.txt')

        # determine the earliest time stamp in the logs as a starting point for the master time line
        earliest_time = min(encoder[0, 0], position[0, 0], trigger[0, 0])
        # get the offsets of every file in milliseconds
        offsets = ((encoder[0, 0] - earliest_time)/1000,
                   (position[0, 0] - earliest_time)/1000,
                   (trigger[0, 0] - earliest_time)/1000)
        # apply offsets to the data so that millisecond-time stamps are aligned, and remove artifact data points
        encoder[:, 0] = encoder[:, 0] + offsets[0]
        encoder[:, 0] = [ceil(x * 1000) / 1000 for x in encoder[:, 0]]
        encoder = np.delete(encoder, 0, 0)
        position[:, 0] = position[:, 0] + offsets[1]
        position[:, 0] = [ceil(x * 1000) / 1000 for x in position[:, 0]]
        position = np.delete(position, [0, 1, position.shape[0]-1], 0)
        trigger[:, 0] = trigger[:, 0] + offsets[2]
        trigger[:, 0] = [ceil(x * 100000) / 100000 for x in trigger[:, 0]]
        trigger = np.delete(trigger, 0, 0)

        # re-arrange frame trigger signal (inverse, only take first trigger stamp)
        trigger[:, 2] = np.invert(trigger[:, 2].astype('bool'))
        trig_blocks = np.split(np.where(trigger[:, 2])[0], np.where(np.diff(np.where(trigger[:, 2])[0]) != 1)[0] + 1)
        # remove artifact before VR start
        trigger[trig_blocks[0], 2] = 0
        # remove trigger duplicates
        for block in trig_blocks[1:]:
            trigger[block[1:], 2] = 0
        # set actual first frame before the first recorded frame (first check if necessary)
        if np.sum(trigger[:, 2]) - frame_list[counter-1] == 0:
            print('Frame count matched, no correction necessary.')
        elif np.sum(trigger[:, 2]) - frame_list[counter-1] == -1:
            print('Imported frame count missed 1, corrected.')
            trigger[trig_blocks[1][0]-67, 2] = 1
        elif np.sum(trigger[:, 2]) - frame_list[counter-1] < -1:
            print(f'{abs(np.sum(trigger[:, 2]) - frame_list[counter-1])} too few frames imported, check files!')
        elif np.sum(trigger[:, 2]) - frame_list[counter-1] > 0:
            print(f'{abs(np.sum(trigger[:, 2]) - frame_list[counter-1])} too many frames imported, check files!')


        ### create the master time line, with one sample every 0.5 milliseconds
        # get maximum and minimum time stamps, rounded to the nearest 0.5 ms step
        min_time = 0.0005 * floor(min(encoder[0, 0], position[0, 0], trigger[0, 0])/0.0005)
        max_time = 0.0005 * ceil(max(encoder[-1, 0], position[-1, 0], trigger[-1, 0])/0.0005)
        master_time = np.arange(start=min_time*10000, stop=(max_time+0.0005)*10000, step=5)/10000
        # create master array with columns [time stamp - position - licks - frame - encoder]
        merge = np.array((master_time, np.zeros(master_time.shape[0]), np.zeros(master_time.shape[0]), np.zeros(master_time.shape[0]), np.zeros(master_time.shape[0]))).T
        for i in [1, 2, 3, 4]:
            merge[:, i] = np.nan

        # go through data and order it into the correct time bin by looking for the time stamp of merge in the other arrays
        # if precise time stamp does not have data, fill in the value of the most recent available time
        last_pos = -10
        last_enc = 0
        for i in range(merge.shape[0]-1):
            if position[merge[i, 0] == position[:, 0], 1].size:     # check if position has the current time stamp
                merge[i, 1] = position[merge[i, 0] == position[:, 0], 1][0] # if yes, fill it in
                last_pos = merge[i, 1]      # update the last available position value
            else:
                merge[i, 1] = last_pos      # if this time point does not have a value, fill in the last value

            if trigger[(merge[i, 0] <= trigger[:, 0]) & (trigger[:, 0] < merge[i + 1, 0]), 1].size:
                merge[i, 2] = max(trigger[(merge[i, 0] <= trigger[:, 0]) & (trigger[:, 0] < merge[i + 1, 0]), 1])
                merge[i, 3] = max(trigger[(merge[i, 0] <= trigger[:, 0]) & (trigger[:, 0] < merge[i + 1, 0]), 2])
            else:
                merge[i, 2] = 0
                merge[i, 3] = 0

            if encoder[merge[i, 0] == encoder[:, 0], 1].size:
                merge[i, 4] = encoder[merge[i, 0] == encoder[:, 0], 1][0]
                last_enc = merge[i, 4]
            else:
                merge[i, 4] = last_enc
            progress(i, merge.shape[0]-1, status='Aligning behavioral data...')

        # clean up file: remove redundant first time stamps, remove last time stamp, reset time stamps
        merge = np.delete(merge, range(np.where(merge[:, 3] == 1)[0][0]), 0)
        merge = np.delete(merge, merge.shape[0]-1, 0)
        merge[:, 0] = merge[:, 0] - merge[0, 0]
        merge[:, 0] = [floor(x * 100000) / 100000 for x in merge[:, 0]]

        # save file (4 decimal places for time (0.5 ms), 2 dec for position, ints for lick, trigger, encoder)
        file_path = curr_path+r'\merged_behavior.txt'
        np.savetxt(file_path, merge, delimiter='\t',
                   fmt=['%.4f', '%.2f', '%1i', '%1i', '%1i'], header='Time\tVR pos\tlicks\tframe\tencoder')
        print(f'Done! \nSaving merged file to {file_path}...')
        counter += 1

#%%

