#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 08/12/2022 15:31
@author: hheise

Gather stats about mice used in all experiments
"""
import pandas as pd
import numpy as np

import hheise_behav
from schema import common_mice, common_exp, common_img
from util import helper

mouse_ids = [33, 38, 41,    # Batch 3
             63, 69,        # Batch 5
             83, 85, 86, 89, 90, 91, 93, 94, 95,  # Batch 7
             108, 110, 111, 112, 113, 114, 115, 116, 120, 121, 122]  # Batch 8

# Get age of mice at first session and microsphere injection
temp_df = []
for mouse in mouse_ids:
    first_sess = (common_exp.Session & 'username="hheise"' & f'mouse_id={mouse}').fetch('day')[0]
    microspheres = (common_mice.Surgery & 'username="hheise"' & f'mouse_id={mouse}' & 'surgery_type="Microsphere injection"').fetch('surgery_date')[0]
    sac = (common_mice.Sacrificed & 'username="hheise"' & f'mouse_id={mouse}').fetch1('date_of_sacrifice')

    first_sess_age = (common_mice.Mouse & 'username="hheise"' & f'mouse_id={mouse}').get_age(first_sess).days
    microsphere_age = (common_mice.Mouse & 'username="hheise"' & f'mouse_id={mouse}').get_age(microspheres).days
    sac_age = (common_mice.Mouse & 'username="hheise"' & f'mouse_id={mouse}').get_age(sac).days
    microsphere_sac = (pd.to_datetime(sac) - pd.to_datetime(microspheres)).days

    temp_df.append(pd.DataFrame(dict(mouse_id=mouse, first_sess=first_sess_age, micro=microsphere_age, death=sac_age,
                                     micro_sac=microsphere_sac), index=(0,)))
ages = pd.concat(temp_df, ignore_index=True)

# Print stats
for col in ['first_sess', 'micro', 'death', 'micro_sac']:
    print(col)
    column = ages[col]
    print(f'\tmin: {column.min()} - max: {column.max()} - mean: {column.mean()} - std: {column.std()}')


# Sex and Strain of mice
sex, strains = (common_mice.Mouse & 'username="hheise"' & f'mouse_id in {helper.in_query(mouse_ids)}').fetch('sex', 'strain')
print(f'Female: {np.sum(sex == "F")}/{len(sex)} ({(np.sum(sex == "F")/len(sex))*100:.2f}%)')
print(f'Male: {np.sum(sex == "M")}/{len(sex)} ({(np.sum(sex == "M")/len(sex))*100:.2f}%)')

for strain in np.unique(strains):
    print(f'{strain}: {np.sum(strains == strain)}/{len(strains)} ({(np.sum(strains == strain)/len(strains))*100:.2f}%)')


# Number of sessions before microsphere injection (learning rate)
temp_df = []
for mouse in mouse_ids:
    microspheres = (common_mice.Surgery & 'username="hheise"' & f'mouse_id={mouse}' & 'surgery_type="Microsphere injection"').fetch('surgery_date')[0].date()

    learning_sess = len(hheise_behav.VRSession & 'username="hheise"' & f'mouse_id={mouse}' & f'day <= "{microspheres}"')-5

    temp_df.append(pd.DataFrame(dict(mouse_id=mouse, learning_sess=learning_sess, index=(0,))))
learning = pd.concat(temp_df, ignore_index=True)
print(f'Baseline recording started after {learning["learning_sess"].mean():.2f} +/- '
      f'{learning["learning_sess"].std():.2f} sessions.')
