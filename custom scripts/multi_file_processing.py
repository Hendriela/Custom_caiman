import place_cell_pipeline as pipe
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from math import ceil, floor
import pickle
from statannot import add_stat_annotation
from ScanImageTiffReader import ScanImageTiffReader
from scipy import stats
from behavior_import import progress
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from glob import glob
import performance_check as performance
from copy import deepcopy
#%%
#r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191121b\N2',
roots = [r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M41\20200318',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M41\20200319',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M41\20200320',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M41\20200321',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M41\20200322',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M41\20200323']

pcf_list = [None] * len(roots)
for idx, root in enumerate(roots):
    pcf_list[idx] = pipe.load_pcf(root)

data = performance.load_performance_data(roots=r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3',
                                         novel=False, norm_date=None)

#%% combine all pcf objects into one big one and plot place cells
root = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data'

def load_total_place_cell_data(root):
    file_list = []
    for step in os.walk(root):
        pcf_files = glob(step[0]+r'\pcf_results*')
        if len(pcf_files) > 0:
            pcf_file = os.path.splitext(os.path.basename(max(pcf_files, key=os.path.getmtime)))[0]
            file_list.append((step[0], pcf_file))
    file_list.reverse()
    print(f'Found {len(file_list)} PCF files. Start loading...')

    bin_avg_act = None
    pc = None

    for file in file_list:

        curr_pcf = pipe.load_pcf(file[0], file[1])

        if bin_avg_act is None:
            bin_avg_act = deepcopy(curr_pcf.bin_avg_activity)
        else:
            idx_offset = bin_avg_act.shape[0]
            try:
                bin_avg_act = np.vstack((bin_avg_act, curr_pcf.bin_avg_activity))
            except ValueError:
                print(f'Couldnt add place cells from {file[0]} because bin number did not add up.')
                continue  # skip the rest of the loop

        if pc is None:
            pc = deepcopy(curr_pcf.place_cells)
        else:
            # Place cell index has to be offset by the amount of cells already in the array
            curr_pc_offset = []
            for i in curr_pcf.place_cells:
                i = list(i)
                i[0] = i[0] + idx_offset
                curr_pc_offset.append(tuple(i))

            pc.extend(curr_pc_offset)

    return bin_avg_act, pc

def plot_total_place_cells(bin_avg_act, place_cells, sort='max', norm=True):

    place_cell_idx = [x[0] for x in place_cells]
    traces = bin_avg_act[place_cell_idx]
    n_neurons = traces.shape[0]

    # Normalize trace array if wanted (every neuron's traces range from 0 to 1)
    if norm:
        for i in range(traces.shape[0]):
            traces[i] = (traces[i] - np.min(traces[i])) / np.ptp(traces[i])

    # sort neurons after different criteria
    bins = []
    if sort == 'max':
        for i in range(n_neurons):
            bins.append((i, np.argmax(traces[i, :])))
    elif sort == 'field':
        for i in range(n_neurons):
            bins.append((i, place_cells[i][1][0][0]))  # get the first index of the first place field
    else:
        print(f'Cannot understand sorting command {sort}.')
        for i in range(n_neurons):
            bins.append((i, i))
    bins_sorted = sorted(bins, key=lambda tup: tup[1])

    # re-sort traces array
    new_permut = [x[0] for x in bins_sorted]
    new_idx = np.empty_like(new_permut)
    new_idx[new_permut] = np.arange(len(new_permut))
    sort_traces = traces[new_permut, :]

    # Plot data
    fig = plt.figure()
    ax = plt.gca()
    # fig.suptitle(f'Place cells of 101 analysed sessions (from 12 mice)', fontsize=16)

    img = ax.pcolormesh(sort_traces, cmap='plasma')

    cbar = fig.colorbar(img, ax=ax, fraction=0.1, )  # draw color bar
    cbar.set_label(r'normalized $\Delta$F/F', labelpad=-10)
    cbar.set_ticks([0, 1])
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.yaxis.label.set_size(15)

    # set axis labels and tidy up graph
    ax.set_xlabel('VR position bin', fontsize=15)
    ax.set_ylabel('# neuron', fontsize=15)
    ax.tick_params(axis='both', labelsize=12)
    ax.invert_yaxis()



        # set x ticks to VR position, not bin number
    # trace_ax[-1, 0].set_xlim(0, traces.shape[1])
    # x_locs, labels = plt.xticks()
    # plt.xticks(x_locs, (x_locs * self.params['bin_length']).astype(int), fontsize=15)
    # plt.sca(trace_ax[-1, 0])
    # plt.xticks(x_locs, (x_locs * self.params['bin_length']).astype(int), fontsize=15)



big_pcf = combine_all_place_cells(root)

#%% spatial information functions

#%%

all_sess_si = []
# for root in roots:
#     if root == r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M25\20191204\N1\pcf_results_nobadtrials.pickle':
#         with open(root, 'rb') as file:
#             pcf = pickle.load(file)
#     else:
#         pcf = pipe.load_pcf(root)

for pcf in pcf_list:

    #data_all = np.load(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191204\N2\deconvolved_PR.npy').T
    # data_all = pcf.cnmf.estimates.F_dff.T
    # position_all = []
    # for trial in range(len(pcf.behavior)):
    #     position_all.append(pcf.behavior[trial][np.where(pcf.behavior[trial][:, 3] == 1), 1])
    # position_all = np.hstack(position_all).T

    spatial_info = pipe.get_spatial_info(pcf.cnmf.estimates.F_dff.T, pcf.behavior, n_bootstrap=0)

    # %% plot spatial info
    pc_label = []
    pc_idx = [x[0] for x in pcf.place_cells]
    pc_rej = [x[0] for x in pcf.place_cells_reject]
    for cell_idx in range(len(spatial_info)):
        if cell_idx in pc_idx:
            pc_label.append('yes')
        # elif cell_idx in pc_rej:
        #     pc_label.append('rejected')
        else:
            pc_label.append('no')

    # add sample size to labels
    no_count = pc_label.count('no')
    acc_count = pc_label.count('yes')
    rej_count = pc_label.count('rejected')
    pc_label = [f'no (n={no_count})' if x == 'no' else x for x in pc_label]
    pc_label = [f'yes (n={acc_count})' if x == 'yes' else x for x in pc_label]
    pc_label = [f'rejected (n={rej_count})' if x == 'rejected' else x for x in pc_label]

    df = pd.DataFrame(data={'SI [bits/sec]': spatial_info, 'Place cell': pc_label, 'dummy': np.zeros(len(spatial_info))})

    # file_dir = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\batch_analysis\spatial information\no_trial_split_bartosformula'
    # mouse = pcf.params['mouse']
    # session = pcf.params['session']
    # file_name = f'spatial_info_{mouse}_{session}.png'
    # fig = plt.figure()
    # plt.title(f'Spatial info {mouse}, {session}')
    # ax = sns.barplot(x='Place cell', y='SI [bits/sec]', data=df)
    # if acc_count > 3: # only perform statistical test if there are at least 4 place cells detected
    #     results = add_stat_annotation(ax, data=df, x='Place cell', y='SI [bits/sec]', text_format='full',
    #                                   box_pairs=[(f'yes (n={acc_count})', f'no (n={no_count})')], test='Mann-Whitney',
    #                                   verbose=2)
    # sns.stripplot(x='Place cell', y='SI [bits/sec]', data=df, linewidth=1)
    # plt.savefig(os.path.join(file_dir, file_name))
    # if root[57:60] == 'M19':
    all_sess_si.append(df)

#%% plot SI over time

# create unified dataframe
all_sess_df = pd.concat(all_sess_si, ignore_index=True)
all_sess_df = all_sess_df.rename(columns={'dummy': 'session'})
# label different sessions
session_dates = ['20191122', '20191125', '20191126', '20191127', '20191203', '20191122', '20191125', '20191126',
                 '20191127', '20191204', '20191205', '20191206', '20191207', '20191208', '20191219']
session_cellcount = [x.shape[0] for x in all_sess_si]
session_label = np.zeros(np.sum(session_cellcount), dtype='object')
last_idx = 0
for i in range(len(session_dates)):
    next_idx = last_idx+session_cellcount[i]
    session_label[last_idx:next_idx] = session_dates[i]
    last_idx = next_idx

all_sess_df['session'] = session_label

# change place cell label (remove n)
pc_labels = list(all_sess_df['Place cell'])                  # temporarily save labels in a list
pc_labels = ['no' if 'no' in x else x for x in pc_labels]    # replace all 'no' labels
pc_labels = ['yes' if 'yes' in x else x for x in pc_labels]  # replace all 'yes' labels
all_sess_df['Place cell'] = pc_labels                        # put edited list back into dataframe

# filter out all place cells with SI = 0

ax = sns.lineplot(x='session', y='SI [bits/sec]', data=all_sess_df)
xlabels = ax.get_xticklabels()
print(xlabels)
ax.set_xticklabels(xlabels, rotation=45, horizontalalignment='right')
plt.tight_layout()

#%% SI barplot

fig = plt.figure()
ax = sns.barplot(x='Place cell', y='SI [bits/sec]', data=all_sess_df)
results = add_stat_annotation(ax, data=all_sess_df, x='Place cell', y='SI [bits/sec]', text_format='full',
                              box_pairs=[(f'yes', f'no')], test='Mann-Whitney',
                              verbose=2)
sns.stripplot(x='Place cell', y='SI [bits/sec]', data=all_sess_df, linewidth=1)

#%% compare pre- and poststroke
#pre_si = pd.concat(all_sess_si[1:6])['SI [bits/sec]']
#post_si = pd.concat(all_sess_si[6:])['SI [bits/sec]']
pre_si = pd.concat(all_sess_si[1:6])
post_d1 = all_sess_si[6]
post_d2 = all_sess_si[7]
post_d3 = all_sess_si[8]
# make list for column labels
pre_post_df = pd.DataFrame(pd.concat([pre_si, post_si], ignore_index=True))
pre_post_label = np.zeros(len(pre_post_df), dtype='object')
pre_post_label[:len(pre_si)] = 'pre'
pre_post_label[len(pre_si):] = 'post'
pre_post_df['Stroke'] = pre_post_label

ax = sns.violinplot(x='Stroke', y='SI [bits/sec]', data=pre_post_df)
results = add_stat_annotation(ax, data=pre_post_df, x='Stroke', y='SI [bits/sec]',
                              box_pairs=[('pre', 'post')], test='Mann-Whitney')
plt.figure();
sns.barplot(x='Stroke', y='SI [bits/sec]', data=pre_post_df)

#%% position decoding
for root in roots:
    if root == r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M25\20191204\N1\pcf_results_nobadtrials.pickle':
        with open(root, 'rb') as file:
            pcf = pickle.load(file)
    else:
        pcf = pipe.load_pcf(root)

    # get data values (dF/F traces) and labels (position)
    dff_all = pcf.cnmf.estimates.F_dff.T
    position_all = []
    for trial in range(len(pcf.behavior)):
        position_all.append(pcf.behavior[trial][np.where(pcf.behavior[trial][:, 3] == 1), 1])
    position_all = np.hstack(position_all).T

    # # only analyse accepted place cells
    # place_cell_idx = [x[0] for x in pcf.place_cells]
    # dff = dff_all[:, place_cell_idx]

    # remove frames/time points where mouse was not moving
    behavior_masks = []
    for trial in pcf.behavior:
        behavior_masks.append(np.ones(int(np.sum(trial[:, 3])), dtype=bool))  # bool list for every frame of that trial
        frame_idx = np.where(trial[:, 3] == 1)[0]  # find sample_idx of all frames
        for i in range(len(frame_idx)):
            if i != 0:
                if np.sum(
                        trial[frame_idx[i - 1]:frame_idx[i], 4]) > -30:  # make the index of the current frame False if
                    behavior_masks[-1][i] = False  # the mouse didnt move much during the frame
            else:
                if trial[0, 4] > -30:
                    behavior_masks[-1][i] = False

    trial_lengths = [int(np.sum(trial)) for trial in behavior_masks]

    behavior_mask = np.hstack(behavior_masks)
    dff = dff_all[behavior_mask]
    position = position_all[behavior_mask]

    #go through the trials and calculate the model once for every trial as test data
    svc_results = []
    for trial in range(len(trial_lengths)):
        if trial != 0:
            min_idx = np.sum(trial_lengths[:trial])
        else:
            min_idx = 0
        max_idx = min_idx + trial_lengths[trial]

        trial_mask = np.zeros(position.shape[0], dtype=bool)
        trial_mask[min_idx:max_idx] = True

        # split dataset into training and testing sets
        train_data = dff[np.where(~trial_mask)[0], :]
        train_label = position[np.where(~trial_mask)[0], :].ravel()
        test_data = dff[np.where(trial_mask)[0], :]
        test_label = position[np.where(trial_mask)[0], :].ravel()

        # make labels int
        # train_label = (train_label*100).astype(int)
        # test_label = (test_label*100).astype(int)

        # make labels int without decimals (bigger categories)
        train_label = train_label.astype(int)
        test_label = test_label.astype(int)

        lab_enc = preprocessing.LabelEncoder()
        encoded_test = lab_enc.fit_transform(test_label)

        clf = LinearSVC(max_iter=10000)
        clf.fit(train_data, train_label)

        pred = clf.predict(test_data)
        score = clf.score(test_data, test_label)

        svc_results.append((pred, score, test_label))

    # %% plot results
    n_trials = len(svc_results)
    ncols = 2
    nrows = ceil(n_trials/ncols)
    fig, ax = plt.subplots(nrows, ncols)
    count = 0
    for row in range(nrows):
        for col in range(ncols):
            if count < len(svc_results):
                ax[row, col].title.set_text('acc. %.1f' % (svc_results[count][1] * 100) + '%')
                ax[row, col].plot(svc_results[count][2], label='orig')
                ax[row, col].plot(svc_results[count][0], label='pred')
                ax[row, col].set_xticks([])
            count += 1
    file_dir = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\batch_analysis\SVC position decoder'
    mouse = pcf.params['mouse']
    session = pcf.params['session']
    file_name = f'SVC_{mouse}_{session}_no_rest.png'
    plt.suptitle(f'SVC {mouse}, {session} no rest')
    plt.savefig(os.path.join(file_dir, file_name))


#%% fraction of transients (crude activity levels)
ratio_df = pd.DataFrame()
for root in roots:
    pcf = pipe.load_pcf(root)

    # combine transient only traces of all neurons into one array
    sess_trans = np.vstack([np.concatenate(x) for x in pcf.session_trans])
    # calculate transient ratio (how many frames consist of significant transients)
    trans_ratio = np.sum(sess_trans, axis=1)/sess_trans.shape[1]

    # create place cell labels
    pc_idx = [x[0] for x in pcf.place_cells]
    pc_labels = np.zeros(len(trans_ratio), dtype=object)
    pc_labels[:] = 'no'
    pc_labels[pc_idx] = 'yes'

    # create session label:
    sess_label = [root[61:69]]*len(trans_ratio)

    # put data into a data frame
    curr_df = pd.DataFrame({'data': trans_ratio, 'session': sess_label, 'pc': pc_labels})

    # merge dataframes
    if ratio_df.empty:
        ratio_df = curr_df
    else:
        ratio_df = pd.concat((ratio_df, curr_df))

#%% correlation coefficients

ratio_df = pd.DataFrame()
for root in roots:
    pcf = pipe.load_pcf(root)

    coef_matrix = np.corrcoef(pcf.cnmf.estimates.F_dff)  # create correlation matrix
    top_idx = np.triu_indices(coef_matrix.shape[0])      # get indices of top triangle
    # coef_matrix[top_idx] = 100                          # set top triangle to arbitrary value for later filtering
    # corrcoef = coef_matrix.flatten()                     # flatten array into a 1D array
    # corrcoef = corrcoef[np.where(corrcoef != 100)]       # remove previously marked indices
    diag_idx = np.diag_indices(coef_matrix.shape[0])
    coef_matrix[diag_idx] = np.nan
    corrcoef = np.nanmean(coef_matrix, axis=1)


    # create place cell labels
    pc_idx = [x[0] for x in pcf.place_cells]
    pc_labels = np.zeros(len(corrcoef), dtype=object)
    pc_labels[:] = 'no'
    pc_labels[pc_idx] = 'yes'

    # create session label:
    sess_label = [root[61:69]]*len(corrcoef)

    # put data into a data frame
    curr_df = pd.DataFrame({'data': corrcoef, 'session': sess_label, 'pc': pc_labels})

    # merge dataframes
    if ratio_df.empty:
        ratio_df = curr_df
    else:
        ratio_df = pd.concat((ratio_df, curr_df))


#%% plot fraction of transients

ratio_df_filtered = ratio_df[ratio_df['session'] != '20191121']
ratio_df_filtered = ratio_df_filtered[ratio_df_filtered['session'] != '20191126']


ax = sns.violinplot(x='session', y='data', hue='pc', data=ratio_df_filtered)
ax = sns.lineplot(x='session', y='data', data=ratio_df_filtered)

# results = add_stat_annotation(ax, data=pre_post_df, x='Stroke', y='SI [bits/sec]',
#                               box_pairs=[('pre', 'post')], test='Mann-Whitney')

def plot_all_place_cells_manual(place_cell_list, save=False, show_neuron_id=False, show_place_fields=True,
                                sort='field', fname='place_cells'):
    """
    Plots all place cells in the data set by line graph and pcolormesh.
    :param save: bool flag whether the figure should be automatically saved in the root and closed afterwards.
    :param show_neuron_id: bool flag whether neuron ID should be plotted next to the line graphs
    :param show_place_fields: bool flag whether place fields should be marked red in the line graph
    :param sort: str, how should the place cells be sorted? 'Max' sorts them for the earliest location of the
    maximum in each trace, 'field' sorts them for the earliest place field.
    :return:
    """
    n_neurons = 0
    trace_list = []
    for pcf_obj in place_cell_list:
        place_cell_idx = [x[0] for x in pcf_obj.place_cells]

        traces = pcf_obj.bin_avg_activity[place_cell_idx]
        n_neurons += traces.shape[0]
    # figure out y-axis limits by rounding the maximum value in traces up to the next 0.05 step
        max_y = 0.05 * ceil(traces.max() / 0.05)
        min_y = 0.05 * floor(traces.min() / 0.05)
        trace_list.append(traces)
    traces = np.vstack(trace_list)

    # sort neurons after different criteria
    bins = []
    for i in range(n_neurons):
        bins.append((i, np.argmax(traces[i, :])))
    bins_sorted = sorted(bins, key=lambda tup: tup[1])

    traces_sort = np.zeros(traces.shape)
    for j in range(len(bins_sorted)):
        traces_sort[j] = traces[bins_sorted[j][0]]

    # remove bad neurons
    bad_traces = np.array([18, 139, 140, 191, 207, 209, 234])
    traces_final = np.delete(traces_sort, bad_traces, axis=0)

    fig = plt.figure()
    img = plt.pcolormesh(traces_final, vmax=max_y, vmin=min_y, cmap='jet')
    ax = plt.gca()
    ax.invert_yaxis()

    # set x ticks to VR position, not bin number
    ax.set_xlim(0, traces.shape[1])
    x_locs, labels = plt.xticks()
    plt.xticks(x_locs, (x_locs * 5).astype(int), fontsize=15)
    plt.yticks(fontsize=15)
    # set axis labels and tidy up graph
    ax.set_xlabel('VR position [cm]', fontsize=18)
    ax.set_ylabel('# neuron', fontsize=18)

    # plot color bar
    # create axis for colorbar
    # axins = inset_axes(ax,
    #                    width="3%",  # width = 5% of parent_bbox width
    #                    height="50%",  # height : 50%
    #                    loc='lower left',
    #                    bbox_to_anchor=(1., 0., 1, 1),
    #                    bbox_transform=ax.transAxes,
    #                    borderpad=0.2)
    cbar = fig.colorbar(img, ax=ax, fraction=.08, label=r'$\Delta$F/F')  # draw color bar
    # cbar = fig.colorbar(img, cax=axins, label=r'$\Delta$F/F')  # draw color bar
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.yaxis.label.set_size(15)

    # align all plots
    trace_fig.subplots_adjust(left=0.1, right=1 - (fraction + 0.05), top=0.9, bottom=0.1)
    plt.show()

    if save:
        plt.savefig(os.path.join(self.params['root'], f'{fname}.png'))
        plt.close()
    # else:
    #     plt.figure()
    #     mouse = self.params['mouse']
    #     session = self.params['session']
    #     network = self.params['network']
    #     plt.title(f'All place cells of mouse {mouse}, session {session}, network {network}', fontsize=16)
    #     if save:
    #         plt.savefig(os.path.join(self.params['root'], f'{fname}.png'))
    #         plt.close()
