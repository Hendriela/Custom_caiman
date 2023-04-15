#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 15/04/2023 15:01
@author: hheise

Functions to prepare data for the siamese network.

"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import registration
from scipy import ndimage
import pandas as pd

from schema import common_match, common_img

#%% FOV shifts with only 2x2 patches (more biologically plausible, less sensitive to noise)


def shift_fov_and_com(sess_key, coms_target):

    def shift_CoMs(CoMs, shift, dims):
        """ Shifts a center-of-mass coordinate point by a certain step size. Caps coordinates at 0 and dims limits
        Args:
            CoMs (iterable): X and Y coordinates of the center of mass.
            shift (iterable): amount by which to shift com, has to be same length as com.
            dims (iterable): dimensions of the FOV, has to be same length as com.
        Returns:
            shifted com
        """
        # shift the CoM by the given amount
        coms_shifted = list(np.full(CoMs.shape[0], np.nan))
        for idx, com in enumerate(CoMs):
            com_shift = [com[0] + shift[0][int(com[0])][int(com[1])], com[1] + shift[1][int(com[0])][int(com[1])]]
            # print(com, shift[idx], com_shift)
            # cap CoM at 0 and dims limits
            com_shift = [0 if x < 0 else x for x in com_shift]
            for coord in range(len(com_shift)):
                if com_shift[coord] > dims[coord]:
                    com_shift[coord] = dims[coord]
            coms_shifted[idx] = tuple(com_shift)
        return np.array(coms_shifted)

    match_keys = sess_key['matched_session'].split('_')
    match_key = dict(username=sess_key['username'], mouse_id=sess_key['mouse_id'], day=match_keys[0],
                     session_num=int(match_keys[1]), motion_id=int(match_keys[2]))

    fov_ref = (common_img.QualityControl & sess_key).fetch1('avg_image')
    fov_match = (common_img.QualityControl & match_key).fetch1('avg_image')
    params = (common_match.CellMatchingParameter & sess_key).fetch1()
    params['fov_shift_patches'] = 1

    img_dim = fov_ref.shape
    patch_size = int(img_dim[0] / params['fov_shift_patches'])
    # Shift maps are a 2D matrix of shape (n_patch, n_patch), with the phase correlation of each patch
    shift_map = np.zeros((2, params['fov_shift_patches'], params['fov_shift_patches']))
    for row in range(params['fov_shift_patches']):
        for col in range(params['fov_shift_patches']):
            # Get a view of the current patch by slicing rows and columns of the FOVs
            curr_ref_patch = fov_ref[row * patch_size:row * patch_size + patch_size,
                             col * patch_size:col * patch_size + patch_size]
            curr_tar_patch = fov_match[row * patch_size:row * patch_size + patch_size,
                             col * patch_size:col * patch_size + patch_size]
            # Perform phase cross correlation to estimate image translation shift for each patch
            patch_shift = registration.phase_cross_correlation(curr_ref_patch, curr_tar_patch, upsample_factor=100,
                                                               return_error=False)
            shift_map[:, row, col] = patch_shift
    # Use scipy's zoom to upscale single-patch shifts to FOV size and get pixel-wise shifts via spline interpolation
    x_shift_map_big = ndimage.zoom(shift_map[0], patch_size, order=3)  # Zoom X and Y shifts separately
    y_shift_map_big = ndimage.zoom(shift_map[1], patch_size, order=3)
    # Further smoothing (e.g. Gaussian) is not necessary, the interpolation during zooming smoothes harsh borders
    shifts = np.stack((x_shift_map_big, y_shift_map_big))

    coms_shifted_target = shift_CoMs(coms_target, shifts, img_dim)

    return shifts, coms_shifted_target


def load_features_and_groundtruth(sess_key, target_key, com_ref, com_tar):
    # Load features and format them
    features_ref = pd.DataFrame((common_match.MatchingFeatures.ROI & sess_key).fetch(as_dict=True))
    features_tar = pd.DataFrame((common_match.MatchingFeatures.ROI & target_key).fetch(as_dict=True))

    features_ref.drop(columns=['contour', 'neighbourhood'], inplace=True)
    features_tar.drop(columns=['contour', 'neighbourhood'], inplace=True)

    # Add CoM
    features_ref['com_x'] = np.array([*com_ref])[:, 0]
    features_ref['com_y'] = np.array([*com_ref])[:, 1]
    features_tar['com_x'] = com_tar[:, 0]
    features_tar['com_y'] = com_tar[:, 1]

    # Split up quadrants into separate features
    features_ref = pd.concat([features_ref,
                              pd.DataFrame(features_ref['neighbours_quadrant'].to_list(),
                                           columns=['quad_ul', 'quad_ur', 'quad_ll', 'quad_lr'])],
                             axis=1).drop(columns=['neighbours_quadrant'])

    features_tar = pd.concat([features_tar,
                              pd.DataFrame(features_tar['neighbours_quadrant'].to_list(),
                                           columns=['quad_ul', 'quad_ur', 'quad_ll', 'quad_lr'])],
                             axis=1).drop(columns=['neighbours_quadrant'])

    # Load ground truth (manually accepted matches)
    ground_truth = pd.DataFrame((common_match.MatchedIndex & sess_key & 'matched_id!=-1').fetch('mask_id', 'matched_id',
                                                                                                as_dict=True))
    return features_ref, features_tar, ground_truth


# Use all matches done recently (except mouse M33, and from one session)
groundtruth_ref_sessions = pd.DataFrame((common_match.MatchedIndex & 'username="hheise"' & 'matched_time > "2023-01-01"'
                                         & 'mouse_id!=33' & 'reverse=0').fetch('username', 'mouse_id', 'day',
                                                                               'session_num', 'motion_id', 'caiman_id',
                                                                               as_dict=True)).drop_duplicates()

ref_features = []
tar_features = []
ground_truth = []
for _, sess in groundtruth_ref_sessions.iterrows():

    matched_sessions = list(np.unique((common_match.MatchedIndex & dict(sess) & 'reverse=0' &
                                       'matched_time > "2023-01-01"').fetch('matched_session')))

    # Only process a single matched session per network to avoid biasing the cells in the reference session
    tar_sess = np.random.choice(matched_sessions)

    for match_sess in matched_sessions:
        # if match_sess == tar_sess:
        # Create primary key dicts for the current session pair
        curr_ref_key = dict(sess, matched_session=match_sess, match_param_id=1)
        curr_tar_key = dict(username=sess['username'], mouse_id=sess['mouse_id'],
                            **common_match.MatchedIndex().string2key(match_sess), match_param_id=1,
                            matched_session=common_match.MatchedIndex().key2title(curr_ref_key))

        # Fetch center of masses
        mask_id_ref, coms_ref = np.vstack((common_img.Segmentation.ROI & curr_ref_key & 'accepted=1').fetch('mask_id', 'com'))
        mask_id_tar, coms_tar_orig = np.vstack(
            (common_img.Segmentation.ROI & curr_tar_key & 'accepted=1').fetch('mask_id', 'com'))

        # Compute new FOV shift without patches, and correct center of masses in the target session
        shift, coms_tar = shift_fov_and_com(sess_key=curr_ref_key, coms_target=coms_tar_orig)

        # Load features, ground truth (accepted matches), and format them
        feat_ref, feat_tar, truth = load_features_and_groundtruth(sess_key=curr_ref_key, target_key=curr_tar_key,
                                                                  com_ref=coms_ref, com_tar=coms_tar)

        ref_features.append(feat_ref)
        tar_features.append(feat_tar)
        ground_truth.append(truth)


# Concatenate all reference and target cells to one large DataFrame, and adjust the ground truth to the new global indices
ref_total = []
tar_total = []
ground_truth_total = []
prev_ref_max_id = 0
prev_tar_max_id = 0

for i, (curr_ref, curr_tar, curr_truth) in enumerate(zip(ref_features, tar_features, ground_truth)):
    #
    # if i == 1:
    #     break

    if i == 0:
        ref_total = curr_ref.copy()
        tar_total = curr_tar.copy()

        ref_total['global_mask_id'] = ref_total['mask_id']
        tar_total['global_mask_id'] = tar_total['mask_id']

        ground_truth_total = curr_truth
        prev_ref = curr_ref

    else:
        if not prev_ref.equals(curr_ref):
            # We have a new network and have to add new reference cells and update the new max global ref ID
            prev_ref_max_id = ref_total['global_mask_id'].max() + 1

            new_ref = curr_ref.copy()
            new_ref['global_mask_id'] = new_ref['mask_id'] + prev_ref_max_id
            ref_total = pd.concat([ref_total, new_ref], ignore_index=True)

        # Adjust the mask IDs: Add the previous highest mask ID to the current IDs to ensure unique IDs across networks
        curr_ground_truth = pd.DataFrame({'mask_id': curr_truth['mask_id'] + prev_ref_max_id,
                                          'matched_id': curr_truth['matched_id'] + prev_tar_max_id})
        ground_truth_total = pd.concat([ground_truth_total, curr_ground_truth], ignore_index=True)

        new_tar = curr_tar.copy()
        new_tar['global_mask_id'] = new_tar['mask_id'] + prev_tar_max_id

        tar_total = pd.concat([tar_total, new_tar], ignore_index=True)

        # if not prev_ref.equals(curr_ref):
        #     # We have a new network and have to add new reference cells
        #     new_ref = curr_ref.copy()
        #     new_ref['global_mask_id'] = new_ref['mask_id'] + prev_ref_max_id

    prev_tar_max_id = tar_total['global_mask_id'].max() + 1


# Save data as CSV files
ref_total.to_csv('.\\custom scripts\\chatgpt_suggestions\\reference_cell_features.csv')
tar_total.to_csv('.\\custom scripts\\chatgpt_suggestions\\target_cell_features.csv')
ground_truth_total.to_csv('.\\custom scripts\\chatgpt_suggestions\\ground_truth.csv')


#
# key = dict(username='hheise', mouse_id=69, day='2021-03-11', session_num=1, motion_id=0,
#            matched_session='2021-03-14_1_0_0', match_param_id=1)
# tar_key = dict(username='hheise', mouse_id=69, day='2021-03-14', session_num=1, motion_id=0)
#
# mask_id_ref, coms_ref = np.vstack((common_img.Segmentation.ROI & key & 'accepted=1').fetch('mask_id', 'com'))
# mask_id_tar, coms_tar_orig = np.vstack((common_img.Segmentation.ROI & tar_key & 'accepted=1').fetch('mask_id', 'com'))
#
# shift, coms_tar = shift_fov_and_com(sess_key=key, coms_target=coms_tar_orig)
#
# feat_ref, feat_tar, truth = load_features_and_groundtruth(sess_key=key, target_key=tar_key,
#                                                           com_ref=coms_ref, com_tar=coms_tar)
#
