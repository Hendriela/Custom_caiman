#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 29/01/2024 15:29
@author: hheise

"""
# filter both cell categorisations and traces, such that the selection of traces and cell categorisations
# are consistent
import pickle
import numpy as np
import os
import pandas as pd

import sys

sys.path.append('../../')


def check_index_consistency(df1, df2):
    return np.all(df1.index == df2.index)


def check_dict_of_dfs_consistency(d1, d2):
    checklist = []
    for df1, df2 in zip(d1.values(), d2.values()):
        checklist.append(check_index_consistency(df1, df2))
    return checklist


def check_cell_coord_consistency(cellcoords, dff):
    difflist = []
    for mouse in dff.keys():
        for sess, col in dff[mouse].items():
            diff = len(cellcoords[mouse][sess].dropna()) - len(col.dropna())
            difflist.append(diff)
        return np.all(np.array(difflist) == 0)


def check_consistency(dict1, dict2):
    nanequal = []
    shapeequal = []
    indequal = []
    for mouse in dict1.keys():
        nanequal.append(np.all(dict1[mouse].isna() == dict2[mouse].isna()))
        shapeequal.append(dict1[mouse].shape == dict2[mouse].shape)
        indequal.append(check_index_consistency(dict1[mouse], dict2[mouse]))
    nanequal_bool = np.all(nanequal)
    shapeequal_bool = np.all(shapeequal)
    indequal_bool = np.all(indequal)

    if np.all([nanequal_bool, shapeequal_bool, indequal_bool]):
        return True
    else:
        return nanequal_bool, shapeequal_bool, indequal_bool


if __name__ == '__main__':
    os.chdir('/home/ga48kuh/NASgroup/labmembers/filippokiessler/wahl-colab')
    scriptname = os.path.basename(__file__)[:-3]
    savestr = f'code/08012024/tracked-cells/outputs/{scriptname}'
    if not os.path.isdir(savestr):
        os.mkdir(savestr)

    pc_division_path = '/preprint/Filippo/neural_data/stability_classes.pkl'
    with open(pc_division_path, 'rb') as file:  # load place cells to filter out those that are not
        pc_classes = pickle.load(file)
    pc_classes = {k: v.reset_index(drop=True) for k, v in pc_classes.items()}

    pc_classes_binary_path = '/preprint/Filippo/neural_data/is_pc.pkl'
    with open(pc_classes_binary_path, 'rb') as file:
        pc_classes_binary = pickle.load(file)
    # reset index of binary pc classes for compatibility
    pc_classes_binary = {k: v.reset_index(drop=True) for k, v in pc_classes_binary.items()}

    dff_path = '/preprint/Filippo/neural_data/dff_tracked_normal.pkl'
    with open(dff_path, 'rb') as file:
        dff = pickle.load(file)
    dff = {k: v.reset_index(drop=True) for k, v in dff.items()}

    decon_path = '/preprint/Filippo/neural_data/decon_tracked_normal.pkl'
    with open(decon_path, 'rb') as file:
        decon = pickle.load(file)
    decon = {k: v.reset_index(drop=True) for k, v in decon.items()}

    coords_path = '/preprint/Filippo/neural_data/cell_coords.pkl'
    with open(coords_path, 'rb') as file:
        coords = pickle.load(file)
    coords = {k: v.reset_index(drop=True) for k, v in coords.items()}

    print(f'check dff consistency with binary place cells: {check_consistency(dff, pc_classes_binary)}')
    print(f'check coordinates consistency with binary place cells: {check_consistency(coords, pc_classes_binary)}')
    print(f'check coordinates consistency with dff traces: {check_consistency(coords, dff)}')
    print(f'check tracked place cells with binary place cells: {check_consistency(pc_classes, pc_classes_binary)}')

