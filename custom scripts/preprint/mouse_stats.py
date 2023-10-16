#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 08/12/2022 15:31
@author: hheise

Gather stats about mice used in all experiments
"""
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn import linear_model

from schema import hheise_behav, hheise_hist, hheise_grouping, hheise_decoder, hheise_pvc, hheise_placecell
from schema import common_mice, common_exp, common_img
from util import helper

mouse_ids = [33, 41,    # Batch 3
             63, 69,        # Batch 5
             83, 85, 86, 89, 90, 91, 93, 95,  # Batch 7
             108, 110, 111, 112, 113, 114, 115, 116, 121, 122]  # Batch 8

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


#%% Learning rate
# Number of sessions before microsphere injection
temp_df = []
for mouse in mouse_ids:
    microspheres = (common_mice.Surgery & 'username="hheise"' & f'mouse_id={mouse}' & 'surgery_type="Microsphere injection"').fetch('surgery_date')[0].date()

    learning_sess = len(hheise_behav.VRSession & 'username="hheise"' & f'mouse_id={mouse}' & f'day <= "{microspheres}"')-5

    temp_df.append(pd.DataFrame(dict(mouse_id=mouse, learning_sess=learning_sess, index=(0,))))
learning = pd.concat(temp_df, ignore_index=True)
print(f'Baseline recording started after {learning["learning_sess"].mean():.2f} +/- '
      f'{learning["learning_sess"].std():.2f} sessions.')

# Slope of VR performance curve
slopes = []
for mouse in mouse_ids:
    surg_date = (common_mice.Surgery & 'username="hheise"' & f'mouse_id={mouse}' &
                 'surgery_type="Microsphere injection"').fetch('surgery_date')[0].date()

    perf = pd.DataFrame((hheise_behav.VRPerformance & 'username="hheise"' & f'mouse_id={mouse}' & f'day <= "{surg_date}"').fetch('mouse_id', 'day', 'si_binned_run', as_dict=True))
    perf['rel_sess'] = np.arange(len(perf), 0, -1)

    # Get slope of linear regression fit from whole data and excluding the baseline imaging sessions
    reg_short = linear_model.LinearRegression().fit(perf.index.to_numpy()[:-5].reshape(-1, 1), perf['si_binned_run'][:-5])
    reg_all = linear_model.LinearRegression().fit(perf.index.to_numpy().reshape(-1, 1), perf['si_binned_run'])
    slopes.append(pd.DataFrame([dict(mouse_id=mouse, slope_learn=reg_short.coef_[0], slope_all=reg_all.coef_[0],
                                     baseline=perf[perf['rel_sess'] < 4]['si_binned_run'].mean(),
                                     max_perf=perf['si_binned_run'].max(),
                                     max_baseline=perf[perf['rel_sess'] < 6]['si_binned_run'].max())]))
slopes = pd.concat(slopes)

#%% Physical exercise (motorized running wheel after stroke)
import statsmodels.api as sm

exercise = [83, 85, 89, 90, 91]
control = [86, 93, 95]

sphere_load = pd.DataFrame((hheise_hist.MicrosphereSummary.Metric() & 'metric_name="spheres"').fetch('mouse_id', 'count', as_dict=True))

df = pd.DataFrame((hheise_grouping.BehaviorGrouping & f'mouse_id in {helper.in_query(mouse_ids)}' &
                   'grouping_id = 0' & 'cluster="coarse"').fetch(as_dict=True))

data = pd.DataFrame(data={**df[['mouse_id', 'early', 'late']]})
data['treatment'] = 0
data.loc[data.mouse_id.isin(exercise), 'treatment'] = 1
# data['treatment'] = pd.Categorical(data['treatment'])
data = pd.merge(data, sphere_load, on='mouse_id')
data = data.rename(columns={'count': 'sphere_load'})
data['sphere_load'] = data['sphere_load'].astype(int)
# data['IsTreated'] = pd.get_dummies(data['treatment'], drop_first=True)

# Create a design matrix with an interaction term
data['sphere_load_Treatment'] = data['sphere_load'] * data['treatment']
data = data.set_index('mouse_id')

# Define the dependent variable (Impairment) and independent variables
X = data[['sphere_load', 'treatment', 'sphere_load_Treatment']]
y = data['early']

# Add a constant to the independent variables (intercept)
X = sm.add_constant(X)

# Fit the multiple linear regression model
model = sm.OLS(y, X).fit()

print(model.summary())

data[['sphere_load', 'early', 'late']].to_clipboard(index=True, header=True)


# Use performance data from first/last poststroke session, like SI performance for behavior matrix
def get_first_last_poststroke(table, attr, restrictions=None, early_day=3, late_day=15, n_last_sessions=3):

    if restrictions is None:
        restrictions = dict(session_num=1)

    data_dfs = []
    for mouse in np.unique(hheise_grouping.BehaviorGrouping().fetch('mouse_id')):

        if mouse == 121:
            continue

        surg_date = (common_mice.Surgery & 'username="hheise"' & f'mouse_id={mouse}' &
                     'surgery_type="Microsphere injection"').fetch('surgery_date')[0].date()
        metric = pd.DataFrame((table & f'mouse_id={mouse}' & restrictions).fetch('day', attr, as_dict=True))

        if len(metric) == 0:
            continue

        metric['rel_day'] = (metric['day'] - surg_date).dt.days

        metric = metric[metric['rel_day'] <= 27]

        # Drop a few outlier sessions (usually last session of a mouse that should not be used)
        if mouse == 83:
            metric = metric.drop(metric[metric['rel_day'] == 27].index)
        elif mouse == 69:
            metric = metric.drop(metric[metric['rel_day'] == 23].index)

        # Make sure that data is sorted chronologically for n_last_sessions to work
        metric = metric.sort_values('rel_day')

        # Early timepoint
        early = metric[(metric['rel_day'] > 0) & (metric['rel_day'] <= early_day)][attr].mean()

        # Late timepoint
        if late_day < 0:
            late = metric[attr].iloc[n_last_sessions:].mean()
        elif (metric['rel_day'] >= late_day).sum() < n_last_sessions:
            # If mouse has less than >n_last_sessions< sessions after late_day,
            # take mean of all available sessions >= late_date
            late = metric[metric['rel_day'] >= late_day][attr].mean()
        else:
            # Otherwise, compute late performance from the last "n_last_sessions" sessions
            late = metric[attr].iloc[-n_last_sessions:].mean()

        data_dfs.append(pd.DataFrame(dict(early=early, late=late), index=[mouse]))
    return pd.concat(data_dfs)


# met = get_first_last_poststroke(hheise_decoder.BayesianDecoderWithinSession, attr='mae_quad', restrictions=dict(bayesian_id=1))
met = get_first_last_poststroke(hheise_pvc.PvcCrossSessionEval, attr='max_pvc',
                                restrictions=dict(circular=0, locations='all'), late_day=-1, n_last_sessions=-1)

met[['early', 'late']].to_clipboard(index=False, header=False)

#%% Validation experiments
data = (hheise_behav.ValidationPerformance & f'task in {helper.in_query(["No pattern", "No tone"])}').plot_task_matrix('blr')

data[['No pattern Prestroke', 'No tone Prestroke']].to_clipboard(header=False)

#%% Test distribution of pauses between lick bursts

licks = pd.DataFrame(hheise_behav.VRSession.VRTrial().fetch('KEY', 'lick', as_dict=True))

zeros_lengths = []
ones_lengths = []
n_bins = 200

for arr in licks.lick:

    groups = [list(g) for k, g in itertools.groupby(arr)]

    zeros_lengths.extend([len(group) for group in groups if (group[0] == 0) and (len(group) < n_bins)])
    ones_lengths.extend([len(group) for group in groups if (group[0] == 1) and (len(group) < n_bins)])

plt.hist(zeros_lengths, bins=n_bins-1, alpha=0.5, label='0s')
plt.hist(ones_lengths, bins=n_bins-1, alpha=0.5, label='1s')
plt.legend()
plt.xlabel('Length of Consecutive 0s/1s')
plt.ylabel('Frequency')
plt.title('Distribution of Lick length (1s) and breaks between Licks (0s)')
