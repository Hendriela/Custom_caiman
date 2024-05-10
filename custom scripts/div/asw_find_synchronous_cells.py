#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 10/05/2024 12:17
@author: hheise

"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load matched dF/F data
with open(r'C:\Users\hheise.UZH\PycharmProjects\Caiman\custom scripts\preprint\Filippo\neural_data\dff.pkl',
          'rb') as file:
    dff = pickle.load(file)


def fill_na(ser):
    new_ser = ser.copy()
    n_frames = len(ser.iloc[np.where(~ser.isna())[0][0]])
    nan_arr = np.array([np.nan]*n_frames)
    new_ser.loc[new_ser.isnull()] = new_ser.loc[new_ser.isnull()].apply(lambda x: nan_arr)
    return new_ser


perc = 95

for mouse_id, mouse_df in dff.items():

    mouse_triplets = []
    for rel_day, series in mouse_df.items():
        triplets = []
        if any(series.isna()):
            to_arr = fill_na(series)
        else:
            to_arr = series.copy()

        corr = np.corrcoef(np.stack(to_arr.values))

        # Find highly correlated pairs
        corr_tril = np.tril(corr, k=-1)
        corr_tril[corr_tril == 0] = np.nan
        threshold = np.nanpercentile(corr_tril, perc)
        highly_correlated_pairs = np.where(corr_tril > threshold)

        # Find combination of three sensors with highest average correlation
        for i, j in zip(highly_correlated_pairs[0], highly_correlated_pairs[1]):
            for k in range(j + 1, len(corr)):
                if k != i and corr[i, k] > threshold and corr[j, k] > threshold:
                    # Calculate mean correlation for the current combination
                    trips = (corr[i, j], corr[i, k], corr[j, k])
                    if not all(np.isnan(trips)):
                        mean_corr = np.nanmean(trips)
                        triplets.append(pd.DataFrame(data=[dict(day=rel_day, i=i, j=j, k=k, mean_corr=mean_corr)]))

        triplet_df = pd.concat(triplets, ignore_index=True)
        triplet_df['mean_corr'] = triplet_df['mean_corr'].round(decimals=12)  # round values to throw out duplicate triplets
        triplet_df_clean = triplet_df.drop_duplicates(subset='mean_corr')
        mouse_triplets.append(triplet_df_clean)
    mouse_triplets_df = pd.concat(mouse_triplets, ignore_index=True).sort_values(by='mean_corr', ascending=False)


# Plot candidates
m_id = 33
cell_ids = [126, 125, 246]  # dataframe row index
days = [-8, -6]             # dataframe column names

m_id = 121
cell_ids = [704, 92, 748]  # dataframe row index
days = [12, 12]             # dataframe column names


fig, ax = plt.subplots(nrows=len(cell_ids), ncols=len(days), sharey='all', sharex='col')
for i, c_id in enumerate(cell_ids):
    for j, d in enumerate(days):
        ax[i, j].plot(dff[m_id].loc[c_id, d])

