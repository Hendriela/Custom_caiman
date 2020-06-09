from standard_pipeline import place_cell_pipeline as pipe
import numpy as np
from sklearn.svm import LinearSVC
from copy import deepcopy
import matplotlib.pyplot as plt

#%% load data from pcf object
root = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191204\N2'
pcf = pipe.load_pcf(root)

#%% get data values (dF/F traces) and labels (position)
dff = pcf.cnmf.estimates.F_dff.T
dff = np.load(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191204\N2\deconvolved_PR.npy').T
#dff = np.nan_to_num(dff)

position = []
for trial in range(len(pcf.behavior)):
    position.append(pcf.behavior[trial][np.where(pcf.behavior[trial][:, 3] == 1), 1])
position = np.hstack(position).T

#%% filter only recognized place cells
dff_all = deepcopy(dff)
place_cell_idx = [x[0] for x in pcf.place_cells]
dff = dff_all[:, place_cell_idx]

#%% remove frames where mouse was stationary
behavior_masks = []
for trial in pcf.behavior:
    behavior_masks.append(np.ones(int(np.sum(trial[:, 3])), dtype=bool))    # bool list for every frame of that trial
    frame_idx = np.where(trial[:, 3] == 1)[0]                               # find sample_idx of all frames
    for i in range(len(frame_idx)):
        if i != 0:
            if np.sum(trial[frame_idx[i-1]:frame_idx[i], 4]) > -30:    # make the index of the current frame False if
                behavior_masks[-1][i] = False                          # the mouse didnt move much during the frame
        else:
            if trial[0, 4] > -30:
                behavior_masks[-1][i] = False

trial_lengths = [int(np.sum(trial)) for trial in behavior_masks]

behavior_mask = np.hstack(behavior_masks)
dff_all = deepcopy(dff)
dff = dff_all[behavior_mask]
position_all = deepcopy(position)
position = position_all[behavior_mask]

#%% bin dff over position into 40 position bins
n_bins = 20
pos_bins = position.copy()
bin_borders = np.linspace(start=-10, stop=110, num=n_bins)
for i in range(len(bin_borders)):
    if i != 0:
        bin_idx = np.where((bin_borders[i-1] <= position) & (position <= bin_borders[i]))[0]    # get idx of bins

        # set all position labels of that bin to a common bin number
        pos_bins[bin_idx] = int(i)

        # bin_blocks = np.split(bin_idx, np.where(np.diff(bin_idx) != 1)[0] + 1)  # separate bins by trial

position = pos_bins.astype(int)

#%% go through the trials and calculate the model once for every trial as test data
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

    #make labels int
    # train_label = (train_label*100).astype(int)
    # test_label = (test_label*100).astype(int)

    # make labels int without decimals
    train_label = train_label.astype(int)
    test_label = test_label.astype(int)

    """
    lab_enc = preprocessing.LabelEncoder()
    encoded_test = lab_enc.fit_transform(test_label)
    """
    clf = LinearSVC(max_iter=100000)
    clf.fit(train_data, train_label)

    pred = clf.predict(test_data)
    score = clf.score(test_data, test_label)

    svc_results.append((pred, score, test_label))
    """
    glm = sm.GLM(train_label, train_data)
    glm_res = glm.fit()
    pred = glm.predict(glm_res.params, test_data)
    score = metrics.accuracy_score(test_label, pred.astype(int))
    glm_results.append((pred, score, test_label))
    
    clf = Perceptron()
    clf.fit(train_data, train_label)
    pred = clf.predict(test_data)
    score = clf.score(test_data, test_label)

    perc_results.append((pred, score, test_label))
    """

#%% plot results
fig, ax = plt.subplots(2, 4)
count = 0
for row in range(2):
    for col in range(4):
        ax[row, col].title.set_text('acc. %.1f' % (svc_results[count][1]*100)+'%')
        ax[row, col].plot(svc_results[count][2], label='orig')
        ax[row, col].plot(svc_results[count][0], label='pred')
        count += 1




#%%

