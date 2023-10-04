#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 03/10/2023 18:24
@author: hheise

"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
import scipy
import matplotlib.pyplot as plt

import hheise_behav
from schema import hheise_decoder, hheise_grouping, hheise_pvc, hheise_hist, common_mice
from util import helper

EXERCISE = [83, 85, 89, 90, 91]

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


def get_aligned_data(table, attr, restrictions=None, norm=False):

    if restrictions is None:
        restrictions = dict(session_num=1)

    # Performance data is loaded from Filippos file and treated separately
    if table == hheise_behav.VRPerformance:
        vr_perf = pd.read_csv(
            r'C:\Users\hheise.UZH\PycharmProjects\Caiman\custom scripts\preprint\Filippo\20230718\vr_performance_aligned.csv')
        vr_perf = vr_perf[(vr_perf['rel_sess'] > -5) & (vr_perf['rel_sess'] <= 27) & (vr_perf['metric'] == attr)]
        vr_perf['group'] = 'control'
        vr_perf.loc[vr_perf['mouse_id'].isin(EXERCISE), 'group'] = 'exercise'

        if norm:
            col_name = 'perf_norm'
        else:
            col_name = 'perf'
        data_df = vr_perf[['mouse_id', 'rel_days', 'group', col_name]]
        data_df = data_df.rename(columns={'rel_days': 'rel_day', col_name: attr})
    else:

        data_dfs = []
        for mouse in np.unique(hheise_grouping.BehaviorGrouping().fetch('mouse_id')):

            if mouse == 121:
                continue

            surg_date = (common_mice.Surgery & 'username="hheise"' & f'mouse_id={mouse}' &
                         'surgery_type="Microsphere injection"').fetch('surgery_date')[0].date()
            metric = pd.DataFrame((table & f'mouse_id={mouse}' & restrictions).fetch('mouse_id', 'day', attr, as_dict=True))

            if len(metric) == 0:
                continue

            rel_days = (metric['day'] - surg_date).dt.days.to_numpy()

            if 3 not in rel_days:
                rel_days[(rel_days == 2) | (rel_days == 4)] = 3
            rel_days[(rel_days == 5) | (rel_days == 6) | (rel_days == 7)] = 6
            rel_days[(rel_days == 8) | (rel_days == 9) | (rel_days == 10)] = 9
            rel_days[(rel_days == 11) | (rel_days == 12) | (rel_days == 13)] = 12
            rel_days[(rel_days == 14) | (rel_days == 15) | (rel_days == 16)] = 15
            rel_days[(rel_days == 17) | (rel_days == 18) | (rel_days == 19)] = 18
            rel_days[(rel_days == 20) | (rel_days == 21) | (rel_days == 22)] = 21
            rel_days[(rel_days == 23) | (rel_days == 24) | (rel_days == 25)] = 24
            if 28 not in rel_days:
                rel_days[(rel_days == 26) | (rel_days == 27) | (rel_days == 28)] = 27

            rel_sess = np.arange(len(rel_days)) - np.argmax(np.where(rel_days <= 0, rel_days, -np.inf))
            pre_mask = (-5 < rel_sess) & (rel_sess < 1)
            rel_days[pre_mask] = np.arange(-np.sum(pre_mask)+1, 1)

            metric['rel_day'] = rel_days
            metric['rel_sess'] = rel_sess
            metric = metric[metric['rel_sess'] >= -4]
            metric = metric[metric['rel_day'] != 1]
            metric = metric[metric['rel_day'] != 2]
            metric = metric[metric['rel_day'] != 4]
            metric = metric[metric['rel_sess'] <= 9]
            # metric = metric[(metric['rel_day'] <= 27)]

            if mouse in EXERCISE:
                metric['group'] = 'exercise'
            else:
                metric['group'] = 'control'

            data_dfs.append(metric[['mouse_id', 'rel_day', 'group', attr]])
        data_df = pd.concat(data_dfs, ignore_index=True)

    sphere_load = pd.DataFrame((hheise_hist.MicrosphereSummary.Metric() &
                                'metric_name="spheres"').fetch('mouse_id', 'count', as_dict=True))

    final_df = pd.merge(data_df, sphere_load, on='mouse_id')
    final_df['count'] = final_df['count'].astype(int)

    return final_df


def fit_linear_regressions(dataset, plot_trend=False, accept_n_missing=0):

    def perform_ols(y_, x_):
        ols_result = sm.OLS(endog=y_, exog=x_).fit()
        return pd.DataFrame([dict(slope=ols_result.params[1], slope_sem=ols_result.bse[1], n=len(y_)-1)])

    attr = dataset.columns[~dataset.columns.isin(['mouse_id', 'rel_day', 'group', 'count'])].values[0]

    lr_results = []
    for day, curr_day in dataset.groupby('rel_day'):

        if len(curr_day) < len(dataset['mouse_id'].unique()) - accept_n_missing:
            continue
        # if len(curr_day) < 10:
        #     continue

        # Fit linear regression on sphere count vs metric for control and exercise groups
        lr_results.append(pd.DataFrame(dict(day=day, group='control',
                                            **perform_ols(y_=curr_day[curr_day['group'] == 'control'][attr],
                                                          x_=sm.add_constant(curr_day[curr_day['group'] == 'control']['count']))
                                            )))
        lr_results.append(pd.DataFrame(dict(day=day, group='exercise',
                                            **perform_ols(y_=curr_day[curr_day['group'] == 'exercise'][attr],
                                                          x_=sm.add_constant(curr_day[curr_day['group'] == 'exercise']['count']))
                                            )))
    lr_result = pd.concat(lr_results, ignore_index=True)
    lr_result = lr_result.set_index(['group', 'day'])       # Make Multiindex to allow for quick group filtering

    # Split control and exercise groups into two separate dataframes, then merge them together again with column prefixes
    lr_control = lr_result.loc['control']
    lr_exercise = lr_result.loc['exercise']
    lr_merged = lr_control.join(lr_exercise, lsuffix="_control", rsuffix="_exercise")

    if plot_trend:
        plt.figure(layout='constrained')
        plt.errorbar(x=lr_control.index, y=lr_control['slope'], yerr=lr_control['slope_sem'], label='control')
        plt.errorbar(x=lr_exercise.index, y=lr_exercise['slope'], yerr=lr_exercise['slope_sem'], label='exercise')
        plt.ylabel('Linear regression slope')
        plt.xlabel('Days after microsphere injection')

    return lr_merged




#%% First random tries, fitting linear models
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

# met = get_first_last_poststroke(hheise_decoder.BayesianDecoderWithinSession, attr='mae_quad', restrictions=dict(bayesian_id=1))
met = get_first_last_poststroke(table=hheise_pvc.PvcCrossSessionEval, attr='max_pvc',
                                restrictions=dict(circular=0, locations='all'))

met[['early', 'late']].to_clipboard(index=False, header=False)


#%% Linear regression test (late max PVC from prism)

X_control = np.array([11, 445, 794, 566, 85, 38, 16, 18, 19, 12, 78, 90, 12, 27, 6]).reshape((-1, 1))
y_control = np.array([0.511818, 0.363356, 0.492931, 0.358073, 0.841046, 0.665236, 0.736778, 0.664153, 0.45356, 0.612145, 0.747605, 0.680901, 0.688531, 0.750302, 0.780298]).reshape((-1, 1))

X_exercise = np.array([22, 223, 32, 461, 7]).reshape((-1, 1))
y_exercise = np.array([0.268023, 0.703182, 0.580707, 0.599976, 0.739563]).reshape((-1, 1))

reg_con = LinearRegression().fit(X_control, y_control)
reg_ex = LinearRegression().fit(X_exercise, y_exercise)

print('Control:', reg_con.coef_[0][0])
print('Exercise:', reg_ex.coef_[0][0])


### Stackoverflow: https://stackoverflow.com/questions/22381497/python-scikit-learn-linear-model-parameter-standard-error
# show your scikit-learn results
print(reg_con.intercept_)
print(reg_con.coef_)

# reproduce this with linear algebra
X_control = pd.DataFrame(X_control, columns=['X_control'])
N = len(X_control)
p = len(X_control.columns) + 1  # plus one because LinearRegression adds an intercept term

X_with_intercept = np.empty(shape=(N, p), dtype=float)
X_with_intercept[:, 0] = 1
X_with_intercept[:, 1:p] = X_control.values

beta_hat = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y_control
print(beta_hat)

# compute standard errors of the parameter estimates
y_hat = reg_con.predict(X_control)
residuals = y_control - y_hat
residual_sum_of_squares = residuals.T @ residuals
sigma_squared_hat = residual_sum_of_squares[0, 0] / (N - p)
var_beta_hat = np.linalg.inv(X_with_intercept.T @ X_with_intercept) * sigma_squared_hat
for p_ in range(p):
    standard_error = var_beta_hat[p_, p_] ** 0.5
    print(f"SE(beta_hat[{p_}]): {standard_error}")

### Replicate with OLS
ols_con = sm.OLS(y_control, sm.add_constant(X_control)).fit()
ols_ex = sm.OLS(y_exercise, sm.add_constant(X_exercise)).fit()

# Calculate the t-statistic and p-value for the difference in slopes
t_statistic = (ols_con.params[1] - ols_ex.params[1]) / np.sqrt(ols_con.bse[1]**2 + ols_ex.bse[1]**2)
degrees_of_freedom = np.min(np.array([len(y_control), len(y_exercise)]) - 2)  # Two parameters were estimated (intercept and slope), so subtract 2
p_value = 2 * (1 - scipy.stats.t.cdf(np.abs(t_statistic), 15))



### Ancova:
data_anc = pd.DataFrame(dict(max_pvc=np.concatenate([y_control, y_exercise]).squeeze(),
                             count=np.concatenate([X_control, X_exercise]).squeeze(),
                             group=[*['control']*len(y_control), *['exercise']*len(y_exercise)]))
formula = 'count ~ group + max_pvc'    # Is the metric different between two groups, accounting for differences in sphere counts
ancova = ols(formula, data_anc).fit()
ancova_table = sm.stats.anova_lm(ancova, typ=2)
print(ancova_table)

#%% Run Prism analysis automatically:
"""
For a given metric (SI performance, Max PVC, decoder mae_quad...)...
    1. Align sessions across mice to have uniform dates
    2. For each session, fit linear regressions to sphere-load vs control and sphere-load vs exercise
    
    The two steps below seem to be difficult to reliably implement in Python, so we take the dataframe from step 2
    and paste it into Prism.
    3. Run one-way ANOVA on the slopes of the four fits to get p-values for slope differences
    4. Plot time course of p-values to find spot where p-values start to decrease --> treatment takes effect
"""

# data = get_aligned_data(table=hheise_pvc.PvcCrossSessionEval, attr='max_pvc',
#                         restrictions=dict(circular=0, locations='all'))
data = get_aligned_data(table=hheise_decoder.BayesianDecoderWithinSession, attr='mae_quad',
                        restrictions=dict(bayesian_id=1))
data = get_aligned_data(table=hheise_behav.VRPerformance, attr='si', norm=True)
result = fit_linear_regressions(dataset=data, plot_trend=False, accept_n_missing=3)
result.to_clipboard(index=True, header=False)





#%% ChatGPT Slope comparison

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# Simulated example data
np.random.seed(0)
n_samples = 100
disease_severity = np.random.rand(n_samples)
control_group = 2 * disease_severity + np.random.randn(n_samples)
treatment_group = 3 * disease_severity + np.random.randn(n_samples)

# Create a DataFrame
data = pd.DataFrame({'Disease_Severity': disease_severity, 'Control_Group': control_group, 'Treatment_Group': treatment_group})

# Fit linear regression models to both groups
X = data['Disease_Severity']
X = sm.add_constant(X)  # Add an intercept term
y_control = data['Control_Group']
y_treatment = data['Treatment_Group']

model_control = sm.OLS(y_control, X).fit()
model_treatment = sm.OLS(y_treatment, X).fit()

# Extract the slopes of the regression lines
slope_control = model_control.params['Disease_Severity']
slope_treatment = model_treatment.params['Disease_Severity']

# Calculate the standard errors for the slopes
se_control = model_treatment.bse['Disease_Severity']
se_treatment = model_treatment.bse['Disease_Severity']

# Calculate the t-statistic and p-value for the difference in slopes
t_statistic = (slope_control - slope_treatment) / np.sqrt(se_control**2 + se_treatment**2)
degrees_of_freedom = len(data) - 2  # Two parameters were estimated (intercept and slope)
p_value = 2 * (1 - stats.t.cdf(np.abs(t_statistic), degrees_of_freedom))

# Set the significance level
alpha = 0.05

# Compare p-value to the significance level
if p_value < alpha:
    print("The slopes are significantly different (p-value < 0.05)")
else:
    print("There is no significant difference in slopes (p-value >= 0.05)")


#%% ChatGPT one-way ANOVA with mean, SEM and N (gives same results as Prisms one-way ANOVA of slope stats

import scipy.stats as stats

means = np.array([-0.000391, -0.0002964, 0.0002109])
sems = np.array([0.000133, 0.0003677, 0.0005357])
sample_sizes = np.array([14, 4, 4])

# Calculate the variances for each group using SEM (SEMs are squared to get variances)
variances = (sems ** 2) * sample_sizes

# Calculate the pooled variance (within-group variance)
pooled_variance = np.sum(variances) / np.sum(sample_sizes - 1)

# Calculate the group means and the overall mean
overall_mean = np.sum(means * sample_sizes) / np.sum(sample_sizes)

# Calculate the between-group sum of squares (SSB)
ssb = np.sum(sample_sizes * (means - overall_mean) ** 2)

# Calculate the within-group sum of squares (SSW)
ssw = np.sum((sample_sizes - 1) * variances)

# Calculate the F-statistic
F_statistic = (ssb / (len(means) - 1)) / (ssw / (np.sum(sample_sizes) - len(means)))

# Calculate the degrees of freedom
df_between = len(means) - 1
df_within = np.sum(sample_sizes) - len(means)

# Calculate the p-value
p_value = 1 - stats.f.cdf(F_statistic, df_between, df_within)


# T-Test for only two slopes --> Not clear how this works, could not find a corresponding value in Prism
# Define the means, standard errors of the mean (SEMs), and sample sizes for both groups
means = np.array([-0.000391, 0.0002109])
sems = np.array([0.000133, 0.0005357])
sample_sizes = np.array([15, 4])

# Calculate the pooled standard error (standard error for the difference in means)
pooled_sem = np.sqrt(np.sum(((sample_sizes - 1) * (sems ** 2)) / (sample_sizes - 2)) / np.sum(1 / sample_sizes))

# Calculate the t-statistic
t_statistic = (means[0] - means[1]) / (pooled_sem * np.sqrt(1 / sample_sizes[0] + 1 / sample_sizes[1]))

# Calculate the degrees of freedom
df = np.sum(sample_sizes - 2)

# Calculate the p-value
p_value = 2 * (1 - stats.t.cdf(np.abs(t_statistic), df))

