#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 30/09/2022 13:50
@author: hheise

"""
import pandas as pd

from schema import common_img

# To be run on local machine (TIFF files on Hard disks)
common_img.MotionCorrection().populate(dict(username='hheise', mouse_id=113, motion_id=0), display_progress=True, reserve_jobs=True, suppress_errors=True)
common_img.MotionCorrection().populate(dict(username='hheise', mouse_id=115, motion_id=0), display_progress=True, reserve_jobs=True, suppress_errors=True)
common_img.MotionCorrection().populate(dict(username='hheise', mouse_id=122, motion_id=0), display_progress=True, reserve_jobs=True, suppress_errors=True)

common_img.QualityControl().populate(dict(username='hheise', mouse_id=113, motion_id=0), display_progress=True, reserve_jobs=True, suppress_errors=True)
common_img.QualityControl().populate(dict(username='hheise', mouse_id=115, motion_id=0), display_progress=True, reserve_jobs=True, suppress_errors=True)
common_img.QualityControl().populate(dict(username='hheise', mouse_id=122, motion_id=0), display_progress=True, reserve_jobs=True, suppress_errors=True)

common_img.Segmentation().populate(dict(username='hheise', mouse_id=113, motion_id=0), display_progress=True, make_kwargs={'save_overviews':True}, reserve_jobs=True, suppress_errors=True)
common_img.Segmentation().populate(dict(username='hheise', mouse_id=115, motion_id=0), display_progress=True, make_kwargs={'save_overviews':True}, reserve_jobs=True, suppress_errors=True)
common_img.Segmentation().populate(dict(username='hheise', mouse_id=122, motion_id=0), display_progress=True, make_kwargs={'save_overviews':True}, reserve_jobs=True, suppress_errors=True)

common_img.Deconvolution().populate({'username': 'hheise', 'decon_id': 1}, display_progress=True, reserve_jobs=True, suppress_errors=True)
common_img.ActivityStatistics().populate({'username': 'hheise'}, display_progress=True, reserve_jobs=True, suppress_errors=True)

# Others in case the VM crashed
common_img.MotionCorrection().populate(dict(username='hheise', mouse_id=108, motion_id=0), display_progress=True, reserve_jobs=True, suppress_errors=True)
common_img.MotionCorrection().populate(dict(username='hheise', mouse_id=110, motion_id=0), display_progress=True, reserve_jobs=True, suppress_errors=True)
common_img.MotionCorrection().populate(dict(username='hheise', mouse_id=111, motion_id=0), display_progress=True, reserve_jobs=True, suppress_errors=True)
common_img.MotionCorrection().populate(dict(username='hheise', mouse_id=112, motion_id=0), display_progress=True, reserve_jobs=True, suppress_errors=True)
common_img.MotionCorrection().populate(dict(username='hheise', mouse_id=114, motion_id=0), display_progress=True, reserve_jobs=True, suppress_errors=True)


common_img.QualityControl().populate(dict(username='hheise', mouse_id=108, motion_id=0), display_progress=True, reserve_jobs=True, suppress_errors=True)
common_img.QualityControl().populate(dict(username='hheise', mouse_id=110, motion_id=0), display_progress=True, reserve_jobs=True, suppress_errors=True)
common_img.QualityControl().populate(dict(username='hheise', mouse_id=111, motion_id=0), display_progress=True, reserve_jobs=True, suppress_errors=True)
common_img.QualityControl().populate(dict(username='hheise', mouse_id=112, motion_id=0), display_progress=True, reserve_jobs=True, suppress_errors=True)
common_img.QualityControl().populate(dict(username='hheise', mouse_id=114, motion_id=0), display_progress=True, reserve_jobs=True, suppress_errors=True)

common_img.Segmentation().populate(dict(username='hheise', mouse_id=108, motion_id=0), display_progress=True, make_kwargs={'save_overviews':True}, reserve_jobs=True, suppress_errors=True)
common_img.Segmentation().populate(dict(username='hheise', mouse_id=110, motion_id=0), display_progress=True, make_kwargs={'save_overviews':True}, reserve_jobs=True, suppress_errors=True)
common_img.Segmentation().populate(dict(username='hheise', mouse_id=111, motion_id=0), display_progress=True, make_kwargs={'save_overviews':True}, reserve_jobs=True, suppress_errors=True)
common_img.Segmentation().populate(dict(username='hheise', mouse_id=112, motion_id=0), display_progress=True, make_kwargs={'save_overviews':True}, reserve_jobs=True, suppress_errors=True)
common_img.Segmentation().populate(dict(username='hheise', mouse_id=114, motion_id=0), display_progress=True, make_kwargs={'save_overviews':True}, reserve_jobs=True, suppress_errors=True)

common_img.Deconvolution().populate(dict(username='hheise', mouse_id=110, decon_id=1), display_progress=True, reserve_jobs=True, suppress_errors=True)
common_img.Deconvolution().populate(dict(username='hheise', mouse_id=110, decon_id=1), display_progress=True, reserve_jobs=True, suppress_errors=True)
common_img.ActivityStatistics().populate({'username': 'hheise'}, display_progress=True, reserve_jobs=True, suppress_errors=True)

# Test flicker
key = dict(username='hheise', mouse_id=112, day='2022-08-27')

# Get frame counts for all trials of the current session
frame_count = (common_img.RawImagingFile & key).fetch('nr_frames')

# Make arrays of the trial's length with the trial's ID and concatenate them to one mask for the whole session
trial_masks = []
for idx, n_frame in enumerate(frame_count):
    trial_masks.append(np.full(n_frame, idx))
trial_mask = np.concatenate(trial_masks)


dirname = r'E:'
dfs = []
for dirpath, folders, file in os.walk(dirname):
    new_files = [os.path.join(dirpath, f) for f in file]
    dfs.append(pd.DataFrame(dict(file=new_files, size=[os.stat(x).st_size for x in new_files])))
df = pd.concat(dfs, ignore_index=True)

df_sort = df.sort_values(by=['size'], ascending=False)
print(df_sort[:10])

# Sort list of files in directory by size
files_sort = [x for _, x in sorted(zip(file_sizes, files))]
file_sizes_sort = [y for y, _ in sorted(zip(file_sizes, files))]
# files = sorted(files, key = lambda x: os.stat(x).st_size)

# Iterate over sorted list of files in directory and
# print them one by one along with size
for i in files[-10:]:
    print(i[1], ' -->', i[0])