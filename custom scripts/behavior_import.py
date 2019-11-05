import numpy as np
from glob import glob
from math import ceil, floor
import sys
import os
import re


def load_file(path):
    """
    Loads files from a directory, with error messages in case there is none or more than 1 file with that name.
    :param path: directory
    :return: loaded file as np.array
    """
    file_path = glob(path)
    if len(file_path) < 1:
        raise Exception(f'No files found at {path}!')
    elif len(file_path) > 1:
        raise Exception(f'File name ambiguous, multiple files found at {path}')
    else:
        return np.loadtxt(file_path[0])


def progress(count, total, status=''):
    """
    Displays an automatically updating progress bar in the console, showing progress of a for-loop.
    :param count: int, current iteration of the loop (current progress)
    :param total: int, maximum iteration of the loop (end point)
    :param status: str, status message displayed next to the progress bar
    :return:
    """
    bar_len = 20
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


# frame_list = [4837, 11174, 709, 3275, 2036, 6274, 7654, 1622, 5475]
# 14 N1 frame_list = [900, 5927, 2430, 1814, 2504, 4624, 6132, 2168, 1953, 2670, 3817]  #todo remove hardcoding

def align_multiple(root, performance_check=False):
    """
    Wrapper for aligning multiple behavioral files. Looks through all subfolders of root for behavioral files.
    If it finds a folder with behavioral files but without merged_behavior.txt, it aligns them.
    If the folder does not contain a .tif file (training without imaging), frame trigger is ignored.
    :param root: string, path of the folder where files are searched
    :param performance_check: boolean flag whether performance should be checked during alignment
    :return: saves merged_behavior.txt for each aligned trial
    """

    for step in os.walk(root):
        if len(step[2]) > 0:   # yes if there are files in the current folder
            if len(glob(step[0] + r'\\Encoder*.txt')) > 0:  # check if the folder has behavioral files
                if 'merged_behavior.txt' not in step[2]:  # check if the trial folder has already been processed
                    if len(glob(step[0] + r'\\*.tif')) > 0:  # check if there is an imaging file for this trial
                        if len(glob(step[0] + r'\\*.mmap')) > 0:    # check if the movie has already been motion corrected
                            print(f'Alignment of files in session {step[0][:-2]} with frame trigger')
                            #align_files(step[0][:-2], imaging=True, performance_check=performance_check) #Todo: remove hard coding of above folder
                        else:
                            print(f'Motion correct .tif movie in {step[0]} before aligning behavioral files.')
                    else:
                        print(f'Alignment of files in session {step[0][:-2]} without frame trigger')
                        # align_files(step[0][:-2], imaging=False, performance_check=performance_check)
            else:
                print(f'No behavioral files in {step[0]}. Alignment not possible.')


def align_files(root, imaging=False, performance_check=False):
    """
    Takes the three behavioral .txt files (encoder, TCP (VR position) and TDT (licks + frame trigger)) and synchronizes
    and aligns them according to the master time stamp provided by Lab View. Data are sampled at the rate of the data
    with the highest sampling rate (TDT, 2 kHz). Missing values of data with lower sampling rate are filled in based on
    their last available value.
    :param root: folder that includes the three behavioral files
    :param imaging: bool flag whether imaging was performed during the session. If yes, TDT frame trigger is compared
                    with frame count of the movie to ensure one trigger signal per frame.
    :param performance_check: bool flag whether the performance should be analyzed (time per trial).
    :return: merged behavioral data saved as 'merged_behavior.txt' in the same folder
    #TODO: adapt so that it looks in multiple subfolders for all behavioral files that have not yet been aligned
    """
    counter = 1

    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split('(\d+)', text)]

    folder_list = glob(root+'/*/')  # find list of trial folders in root directory
    folder_list.sort(key=natural_keys)

    if performance_check:
        trial_times = []

    for folder in folder_list:
        # get frame count of the current trial from memmap file name
        if imaging:
            frame_count = int(glob.glob(folder+'*.mmap')[0].split('_')[-2])

        if performance_check:
            root = folder.split('\\')[0]
            save_file = os.path.join(root, r'performance.txt')
            with open(save_file, 'w') as f:
                out = f.write('Performance for the current session:')

        print(f'\nNow processing trial {counter} of {len(folder_list)}: Folder {folder}...')
        # load the three files (encoder (running speed), TCP (VR position) and TDT (licking + frame trigger))
        encoder = load_file(os.path.join(folder, r'Encoder*.txt'))
        position = load_file(os.path.join(folder, r'TCP*.txt'))
        trigger = load_file(os.path.join(folder, r'TDT*.txt'))

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

        if imaging:
            # re-arrange frame trigger signal (inverse, only take first trigger stamp)
            trigger[:, 2] = np.invert(trigger[:, 2].astype('bool'))
            trig_blocks = np.split(np.where(trigger[:, 2])[0], np.where(np.diff(np.where(trigger[:, 2])[0]) != 1)[0] + 1)
            # remove artifact before VR start
            trigger[trig_blocks[0], 2] = 0
            # remove trigger duplicates
            for block in trig_blocks[1:]:
                trigger[block[1:], 2] = 0
            # set actual first frame before the first recorded frame (first check if necessary)
            if np.sum(trigger[:, 2]) - frame_count == 0:
                print('Frame count matched, no correction necessary.')
            elif np.sum(trigger[:, 2]) - frame_count <= -1:
                missing_frames = int(np.sum(trigger[:, 2]) - frame_count)
                if trig_blocks[1][0] - missing_frames*67 > 0:
                    trigger[trig_blocks[1][0] - missing_frames*67, 2] = 1
                    print(f'Imported frame count missed {int(abs(missing_frames))}, corrected.')
                else:
                    print(f'{int(abs(missing_frames))} too few frames imported from TDT, could not be corrected.')
            elif np.sum(trigger[:, 2]) - frame_count > 0:
                print(f'{int(abs(np.sum(trigger[:, 2]) - frame_count))} too many frames imported from TDT, check files!')


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

            # look at adjacent time points, if they are less than 0.5 ms apart, take the maximum (1 if there was a frame during that time)
            if trigger[(merge[i, 0] <= trigger[:, 0]) & (trigger[:, 0] < merge[i + 1, 0]), 1].size:
                merge[i, 2] = max(trigger[(merge[i, 0] <= trigger[:, 0]) & (trigger[:, 0] < merge[i + 1, 0]), 1])
                if imaging: # not needed without imaging during this session
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
        if imaging:
            merge = np.delete(merge, range(np.where(merge[:, 3] == 1)[0][0]), 0)
        merge = np.delete(merge, merge.shape[0]-1, 0)
        merge[:, 0] = merge[:, 0] - merge[0, 0]
        merge[:, 0] = [floor(x * 100000) / 100000 for x in merge[:, 0]]

        if not imaging:
            merge[:, 3] = 0

        if performance_check:
            print(f'Trial {counter} was {merge[-1, 0]} s long.')
            trial_times.append(merge[-1, 0])

            with open(save_file, 'a') as text_file:
                out = text_file.write(f'Trial {counter} was {merge[-1, 0]} s long.')

        # save file (4 decimal places for time (0.5 ms), 2 dec for position, ints for lick, trigger, encoder)
        file_path = os.path.join(folder, r'merged_behavior.txt')
        np.savetxt(file_path, merge, delimiter='\t',
                   fmt=['%.4f', '%.2f', '%1i', '%1i', '%1i'], header='Time\tVR pos\tlicks\tframe\tencoder')
        print(f'Done! \nSaving merged file to {file_path}...')
        counter += 1

    if performance_check:
        print(f'\nPerformance parameters:\nAverage trial time: {np.mean(trial_times)}s'
              f'\nFastest trial: {np.min(trial_times)}s (trial {np.argmin(trial_times)+1})'
              f'\nLongest trial: {np.max(trial_times)}s (trial {np.argmax(trial_times)+1})')
        with open(save_file, 'a') as f:
            out = f.write(f'\nPerformance parameters:\nAverage trial time: {np.mean(trial_times)}s'
                          f'\nFastest trial: {np.min(trial_times)}s (trial {np.argmin(trial_times)+1})'
                          f'\nLongest trial: {np.max(trial_times)}s (trial {np.argmax(trial_times)+1})')
#%%

