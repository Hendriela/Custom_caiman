#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 22/06/2023 11:33
@author: hheise

"""
import pickle
import os

from schema import hheise_placecell, common_match

#%% Load matched/fetched data
dir = r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Data\analysis\matched_data'
with open(os.path.join(dir, 'is_pc.pickle'), "r") as output_file:
    is_pc = pickle.load(output_file)
with open(os.path.join(dir, 'pfs.pickle'), "r") as output_file:
    pfs = pickle.load(output_file)
with open(os.path.join(dir, 'spatial_maps.pickle'), "r") as output_file:
    spatial_maps = pickle.load(output_file)
with open(os.path.join(dir, 'spat_dff_maps.pickle'), "r") as output_file:
    spat_dff_maps = pickle.load(output_file)

#%% Make a place-cell-heatmap from deficit mice (41+69), recovery mice (85+90) and sham mice (115+122)

# M41

