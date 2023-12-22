#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 28/11/2023 11:19
@author: hheise

"""

import numpy as np
import pandas as pd
import os
from schema import hheise_hist, common_hist
from scipy import stats
from sklearn import linear_model

os.chdir('.\\custom scripts\\preprint\\Filippo')

victor = [11, 13, 14, 19, 20, 23, 26, 32, 33, 35, 36]
no_deficit = [93, 91, 95]
no_deficit_flicker = [111, 114, 116]
recovery = [33, 83, 85, 86, 89, 90, 113]
deficit_no_flicker = [41, 63, 69, 121]
deficit_flicker = [108, 110, 112]
sham_injection = [115, 122]
mice = [*no_deficit, *no_deficit_flicker, *recovery, *deficit_no_flicker, *deficit_flicker, *sham_injection, *victor]

columns = ['spheres', 'spheres_rel', 'spheres_lesion', 'spheres_lesion_rel', 'lesion_vol', 'lesion_vol_rel',
           'spheres_extrap', 'spheres_rel_extrap', 'spheres_lesion_extrap', 'spheres_lesion_rel_extrap',
           'region', 'mouse_id', 'username']


def clean_dataframe(old_df, mice_ids):

    # Only keep mice that are included in mice_ids
    new_df = old_df[old_df['mouse_id'].isin(mice_ids)]
    new_rows = []

    # Find mice that don't have a value for all regions and enter dummy data (zeros)
    for (region, mouse, user), sub_df in new_df.groupby(['region', 'mouse_id', 'username']):
        if mouse not in new_df[(new_df['region'] == region) & (new_df['username'] == user)]['mouse_id'].values:
            if mouse < 108:
                dummy_data = np.zeros(10)
            else:
                dummy_data = np.array([0, 0, np.nan, np.nan, np.nan, np.nan, 0, 0, np.nan, np.nan])
            dummy_data = np.append(dummy_data, [region, mouse, user])

            new_rows.append(pd.DataFrame(dummy_data.reshape(1, -1), columns=list(new_df)))

    if len(new_rows) > 0:
        new_df = pd.concat([new_df, pd.concat(new_rows, ignore_index=True)], ignore_index=True)

    new_df = new_df[new_df['mouse_id'] != '63']
    new_df = new_df[new_df['mouse_id'] != '69']
    new_df_sort = new_df.sort_values(by=['region', 'mouse_id'], ignore_index=True)

    conversion = {col: (str if col in ['region', 'mouse_id', 'username'] else float) for col in new_df_sort.columns}
    new_df_sort = new_df_sort.astype(conversion)
    new_df_sort['mouse_id'] = new_df_sort['mouse_id'].astype(int)

    return new_df_sort


def compute_relatives(dataframe, col):
    mouse_sums = dataframe.groupby('mouse_id').agg({col: 'sum'})
    dataframe[f'{col}_rel'] = dataframe.apply(lambda x: x[col]/mouse_sums.loc[x['mouse_id']][col]*100, axis=1)
    return dataframe

#%%
df = hheise_hist.Microsphere().get_structure_groups(grouping={'hippocampus': ['HPF'],
                                                              'white_matter': ['fiber tracts'],
                                                              'striatum': ['STR'],
                                                              'thalamus': ['TH'],
                                                              'neocortex': ['Isocortex'],
                                                              },
                                                    columns=columns, lesion_stains=['map2', 'gfap', 'auto'])
df = clean_dataframe(old_df=df, mice_ids=mice)[['spheres', 'lesion_vol', 'region', 'mouse_id', 'username']]

# Compute relative sphere and lesion volume
df = compute_relatives(df, 'spheres')
df = compute_relatives(df, 'lesion_vol')

# Exclude mice with less than 15 spheres
total_spheres = df.groupby(['username', 'mouse_id']).agg({'spheres': 'sum'}).reset_index().rename(columns={'spheres':'total_mouse_spheres'})
df = pd.merge(df, total_spheres, on=['username', 'mouse_id'])
df_filt = df[df['total_mouse_spheres'] >= 15]

df_filt.pivot(index='region', columns='mouse_id', values='lesion_vol_rel').to_clipboard(index=False, header=False)

#%% Performance group - sphere count correlation

data = pd.DataFrame({'mouse_id': np.array([33, 41, 63, 69, 83, 85, 86, 89, 90, 91, 93, 95, 108, 110, 111, 112, 113, 114, 115, 116, 122]),
                     'class': np.array([0, 2, 2, 2, 0, 1, 1, 1, 1, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0]),
                     'spheres': np.array([57.6435, 3259.67, 4564.95, 3787.27, 105.84, 1227.2, 401.981, 147.385, 2081.68,
                                          30.2646, 164.044, 71.252, 55.616, 61.8407, 38.9562, 224.147, 249.409, 323.075,
                                          41.1545, 84.8064, 18.5548])})

corr = stats.spearmanr(data['class'], data['spheres'])

linreg = linear_model.LinearRegression().fit(data['class'].to_numpy().reshape(-1, 1), data['spheres'].to_numpy().reshape(-1, 1))
r_squared = linreg.score(data['class'].to_numpy().reshape(-1, 1), data['spheres'].to_numpy().reshape(-1, 1))

import statsmodels.api as sm
alpha = 0.05  # 95% confidence interval
lr = sm.OLS(data['spheres'], sm.add_constant(data['class'])).fit()
conf_interval = lr.conf_int(alpha)


