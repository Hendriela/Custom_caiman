import place_cell_pipeline as pipe
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from math import ceil
import pickle

roots = [r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191121b\N2',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191205\N2',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M25\20191121b\N1',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M25\20191204\N1\pcf_results_nobadtrials.pickle',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M18\20191121b\N1',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M18\20191204\N1']

#%% spatial information
for root in roots:
    if root == r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M25\20191204\N1\pcf_results_nobadtrials.pickle':
        with open(root, 'rb') as file:
            pcf = pickle.load(file)
    else:
        pcf = pipe.load_pcf(root)

    pcf.import_behavior_and_align_traces(remove_resting_frames=True)
    # pcf.create_transient_only_traces()
    # pcf.find_place_cells()
    # if len(pcf.place_cells) > 0:
    #     pcf.plot_all_place_cells(save=False, show_neuron_id=True)
    # pcf.save('pcf_results_no_resting')
    spatial_info = pcf.get_spatial_information(trace='S')
    pc_label = []
    pc_idx = [x[0] for x in pcf.place_cells]
    pc_rej = [x[0] for x in pcf.place_cells_reject]
    for cell_idx in range(len(spatial_info)):
        if cell_idx in pc_idx:
            pc_label.append('accepted')
        elif cell_idx in pc_rej:
            pc_label.append('rejected')
        else:
            pc_label.append('no')
    df = pd.DataFrame(data={'SI': spatial_info, 'Place cell': pc_label, 'dummy': np.zeros(len(spatial_info))})
    plt.figure()
    sns.stripplot(x='dummy', y='SI', data=df, hue='Place cell')
    file_dir = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\batch_analysis\spatial information'
    mouse = pcf.params['mouse']
    session = pcf.params['session']
    file_name = f'spatial_info_{mouse}_{session}_no_rest.png'
    plt.title(f'Spatial info {mouse}, {session}')
    plt.savefig(os.path.join(file_dir, file_name))

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


#%%

