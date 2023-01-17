from standard_pipeline import behavior_import as behavior
import numpy as np
from collections import Counter

path_undist = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\labview_logging_test\logging_test_300frames_undisturbed\TDT TASK DIG-IN_20191127_101336.txt'
path_window = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\labview_logging_test\logging_test_300frames_windowmoved\TDT TASK DIG-IN_20191127_101515.txt'
path_server = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\labview_logging_test\logging_test_300frames_servertransfer\TDT TASK DIG-IN_20191127_101657.txt'
trig_path = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M22\20191206\N1\7\TDT TASK DIG-IN_20191206_121510.txt'
enc_path = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M22\20191206\N1\7\Encoder data20191206_121510.txt'
pos_path = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M22\20191206\N1\7\TCP read data20191206_121510.txt'

merged_path = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191121b\N2\2\merged_behavior.txt'
merged = np.loadtxt(merged_path, delimiter='\t')

pos = behavior.load_file(pos_path)
pos = np.delete(pos, 0, 0)
enc = behavior.load_file(enc_path)
enc = np.delete(enc, 0, 0)

real = behavior.load_file(path_real)
real = np.delete(real, 0, 0)
undist = behavior.load_file(path_undist)
undist = np.delete(undist, 0, 0)
window = behavior.load_file(path_window)
window = np.delete(window, 0, 0)
server = behavior.load_file(path_server)
server = np.delete(server, 0, 0)

#%%
undist_blocks = np.split(np.where(undist[:, 2])[0], np.where(np.diff(np.where(undist[:, 2])[0]) != 1)[0] + 1)
window_blocks = np.split(np.where(window[:, 2])[0], np.where(np.diff(np.where(window[:, 2])[0]) != 1)[0] + 1)
server_blocks = np.split(np.where(server[:, 2])[0], np.where(np.diff(np.where(server[:, 2])[0]) != 1)[0] + 1)

real_inv = np.invert(real[:, 2].astype('bool'))

merge_blocks = np.split(np.where(trigger[:, 2])[0], np.where(np.diff(np.where(trigger[:, 2])[0]) != 1)[0] + 1)
real_inv_blocks = np.split(np.where(real_inv)[0], np.where(np.diff(np.where(real_inv)[0]) != 1)[0] + 1)

undist_rev = np.invert(undist[:, 2].astype('bool'))
undist_rev_blocks = np.split(np.where(undist_rev[:,2])[0], np.where(np.diff(np.where(undist_rev[:,2])[0]) != 1)[0] + 1)

frame_times = []
for frame in trig_blocks:
    frame_times.append(len(frame))

counter = Counter(frame_times)
print(counter)

#%% check sample rate
sample_time = []
for i in range(trigger.shape[0]-1):
    sample_time.append((trigger[i+1, 0]-trigger[i, 0])*1000)

#%%

bin_frame_count = np.zeros((pcf.params['n_bins'], pcf.params['n_trial']), 'int')
for trial in range(len(behavior)):  # go through vr data of every trial and prepare it for analysis

    # bin data in distance chunks
    bin_borders = np.linspace(-10, 110, pcf.params['n_bins'])
    idx = np.digitize(merged[:, 1], bin_borders)  # get indices of bins

    # check how many frames are in each bin
    for i in range(pcf.params['n_bins']):
        bin_frame_count[i, trial] = np.sum(merged[np.where(idx == i + 1), 3])

# double check if number of frames are correct
for i in range(len(pcf.params['frame_list'])):
    frame_list_count = pcf.params['frame_list'][i]
    if frame_list_count != np.sum(bin_frame_count[:, i]):
        print(
            f'Frame count not matching in trial {i + 1}: Frame list says {frame_list_count}, import says {np.sum(bin_frame_count[:, i])}')

#%%
import numpy as np
path = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch7\M93\20210720\TDT TASK DIG-IN_20210720_142150_REMOVE FIRST BIT OF FRAME TRIGGER.txt'
data = np.loadtxt(path)

np.savetxt(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch7\M93\20210720\TDT TASK DIG-IN_20210720_142150.txt', data[1:],
           fmt=['%.5f', '%.5f', '%.5f'], header='142150624.00000\t142150688.72499\t3.00000')


#%% Check disk usage
import os

def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                try:
                    total_size += os.path.getsize(fp)
                except FileNotFoundError:
                    print("Skipped file {}".format(fp))

    return total_size



wahl_size = get_size('W:\\Neurophysiology-Storage1\\Wahl')
print(get_size(), 'bytes')


#%% Test motion correction parameters for M63
from schema import common_img
key = dict(mouse_id=63, day="2021-03-05")

common_img.ScanInfo().populate(key)

common_img.MotionParameter & key

common_img.MotionCorrection().populate(dict(**key, motion_id=0))
common_img.MotionCorrection().populate(dict(**key, motion_id=1))
common_img.MotionCorrection().populate(dict(**key, motion_id=2))

(common_img.MotionCorrection() & dict(**key, motion_id=0)).export_tif(target_folder=r'F:\Batch5\M63\20210305\motion_id_0', remove_after=True)
(common_img.MotionCorrection() & dict(**key, motion_id=1)).export_tif(target_folder=r'F:\Batch5\M63\20210305\motion_id_1', remove_after=True)
(common_img.MotionCorrection() & dict(**key, motion_id=2)).export_tif(target_folder=r'F:\Batch5\M63\20210305\motion_id_2', remove_after=True)
