from caiman.source_extraction import cnmf
import place_cell_pipeline as pipe
import behavior_import as behavior
import place_cell_class as pc
import caiman as cm
import matplotlib.pyplot as plt
import os
import performance_check as performance
import numpy as np
import seaborn as sns

"""
Complete pipeline for place cell analysis, from motion correction to place cell finding

Condition before starting:
Tiff and behavioral files (encoder, position, frames/licks) have to be in separate folders for each trial, and trial
folders have to be grouped in one folder per field of view. Several FOVs can be grouped in a folder per session.
"""
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)
#%% Set parameters

# dataset dependent parameters
fr = 30  # imaging rate in frames per second
decay_time = 0.4  # length of a typical transient in seconds (0.4)
dxy = (1.66, 1.52)  # spatial resolution in x and y in (um per pixel) [(1.66, 1.52) for 1x, (0.83, 0.76) for 2x]
# note the lower than usual spatial resolution here
max_shift_um = (50., 50.)  # maximum shift in um
patch_motion_um = (100., 100.)  # patch size for non-rigid correction in um

# motion correction parameters
pw_rigid = True  # flag to select rigid vs pw_rigid motion correction
# maximum allowed rigid shift in pixels
max_shifts = [int(a / b) for a, b in zip(max_shift_um, dxy)]
# start a new patch for pw-rigid motion correction every x pixels
strides = tuple([int(a / b) for a, b in zip(patch_motion_um, dxy)])
# overlap between patches (size of patch in pixels: strides+overlaps)
overlaps = (12, 12)
# maximum deviation allowed for patch with respect to rigid shifts
max_deviation_rigid = 3

mc_dict = {
    'fnames': None,
    'fr': fr,
    'decay_time': decay_time,
    'dxy': dxy,
    'pw_rigid': pw_rigid,
    'max_shifts': max_shifts,
    'strides': strides,
    'overlaps': overlaps,
    'max_deviation_rigid': max_deviation_rigid,
    'border_nan': 'copy',
    'n_processes': n_processes
}

opts = cnmf.params.CNMFParams(params_dict=mc_dict)


#%% Set working directory

root = pipe.set_file_paths()

#%% Align behavioral data
root = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M41\20200321'
behavior.align_behavior(root, performance_check=False, verbose=False, overwrite=False)

# evaluate behavior
mouse_dir_list = [r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M18',
              r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19',
              r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M22',
              r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M25']
date_0 = ['20191203', '20191204', '20191204', '20191204']

performance.multi_mouse_performance(mouse_list, precise_duration=False, separate_zones=True, date_0=date_0)

#%% Perform motion correction

roots = [r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M32\20200322',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M41\20200322',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M33\20200322',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M35\20200322',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M37\20200322',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M38\20200322',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M39\20200322',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M40\20200322',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M32\20200323',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M41\20200323',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M33\20200323',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M35\20200323',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M37\20200323',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M38\20200323',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M39\20200323',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M40\20200323']

for root in roots:
    motion_file, dview = pipe.motion_correction(root, opts, dview, remove_f_order=True, remove_c_order=True,
                                                get_images=True, overwrite=True)




#%% CaImAn source extraction
mmap_file, images = pipe.load_mmap(root)

#%% whole pipeline

# set parameters
# dataset dependent parameters
fr = 30  # imaging rate in frames per second
decay_time = 0.4  # length of a typical transient in seconds (0.4)
dxy = (1.66, 1.52)  # spatial resolution in x and y in (um per pixel) [(1.66, 1.52) for 1x, (0.83, 0.76) for 2x]
# note the lower than usual spatial resolution here
max_shift_um = (50., 50.)  # maximum shift in um
patch_motion_um = (100., 100.)  # patch size for non-rigid correction in um

# motion correction parameters
pw_rigid = True  # flag to select rigid vs pw_rigid motion correction
# maximum allowed rigid shift in pixels
max_shifts = [int(a / b) for a, b in zip(max_shift_um, dxy)]
# start a new patch for pw-rigid motion correction every x pixels
strides = tuple([int(a / b) for a, b in zip(patch_motion_um, dxy)])
# overlap between pathes (size of patch in pixels: strides+overlaps)
overlaps = (12, 12)
# maximum deviation allowed for patch with respect to rigid shifts
max_deviation_rigid = 3

mc_dict = {
    'fnames': None,
    'fr': fr,
    'decay_time': decay_time,
    'dxy': dxy,
    'pw_rigid': pw_rigid,
    'max_shifts': max_shifts,
    'strides': strides,
    'overlaps': overlaps,
    'max_deviation_rigid': max_deviation_rigid,
    'border_nan': 'copy',
    'n_processes': n_processes
}

opts = cnmf.params.CNMFParams(params_dict=mc_dict)

# extraction parameters
p = 1  # order of the autoregressive system
gnb = 2  # number of global background components (3)
merge_thr = 0.75  # merging threshold, max correlation allowed (0.86)
rf = 25
# half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
stride_cnmf = 20  # amount of overlap between the patches in pixels (20)
K = 20  # number of components per patch (10)
gSig = [5, 5]  # expected half-size of neurons in pixels
# initialization method (if analyzing dendritic data using 'sparse_nmf')
method_init = 'greedy_roi'
ssub = 2  # spatial subsampling during initialization
tsub = 2  # temporal subsampling during intialization

opts_dict = {'fnames': None,
             'nb': gnb,
             'rf': rf,
             'K': K,
             'gSig': gSig,
             'stride': stride_cnmf,
             'method_init': method_init,
             'rolling_sum': True,
             'merge_thr': merge_thr,
             'only_init': True,
             'ssub': ssub,
             'tsub': tsub}
opts = opts.change_params(params_dict=opts_dict)

# evaluation parameters
min_SNR = 7  # signal to noise ratio for accepting a component (default 2)
SNR_lowest = 4
rval_thr = 0.95  # space correlation threshold for accepting a component (default 0.85)
cnn_thr = 0.99  # threshold for CNN based classifier (default 0.99)
cnn_lowest = 0.1  # neurons with cnn probability lower than this value are rejected (default 0.1)

opts_dict = {'decay_time': decay_time,
             'SNR_lowest': SNR_lowest,
             'min_SNR': min_SNR,
             'rval_thr': rval_thr,
             'use_cnn': True,
             'min_cnn_thr': cnn_thr,
             'cnn_lowest': cnn_lowest}
opts = opts.change_params(params_dict=opts_dict)

# place cell parameters
pcf_params = {'root': None,                  # main directory of this session
          'trans_length': 0.5,           # minimum length in seconds of a significant transient
          'trans_thresh': 4,             # factor of sigma above which a transient is significant
          'bin_length': 5,               # length in cm VR distance of each bin in which to group the dF/F traces (has to be divisor of track_length
          'bin_window_avg': 3,           # sliding window of bins (left and right) for trace smoothing
          'bin_base': 0.25,              # fraction of lowest bins that are averaged for baseline calculation
          'place_thresh': 0.25,          # threshold of being considered for place fields, calculated
                                         # from difference between max and baseline dF/F
          'min_pf_size': 15,             # minimum size in cm for a place field (should be 15-20 cm)
          'fluo_infield': 7,             # factor above which the mean DF/F in the place field should lie compared to outside the field
          'trans_time': 0.2,             # fraction of the (unbinned!) signal while the mouse is located in
                                         # the place field that should consist of significant transients
          'track_length': 400,           # length in cm of the virtual reality corridor
          'split_size': 50}              # size in frames of bootstrapping segments

#%%
roots = [r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M18\20191121b\N1',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M18\20191122a\N1',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M18\20191125\N1',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M18\20191203b\N1',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M18\20191204\N1',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M18\20191205\N1',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M18\20191206\N1',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M18\20191207\N1',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M18\20191219\N1']


for root in roots:
    pipe.whole_caiman_pipeline_mouse(root, opts, pcf_params, dview, make_lcm=True, network='N1')

#%% separate functions
cnm = pipe.run_source_extraction(images, opts, dview=dview)
#if images.shape[0] > 40000:
#    lcm = pipe.get_local_correlation(images[::2])
#else:
lcm = pipe.get_local_correlation(images)
cnm.estimates.Cn = lcm
cnm.estimates.plot_contours(img=lcm, display_numbers=False)
plt.savefig(os.path.join(root, 'all_components.png'))
plt.close()

#%% serial evaluation
root = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M32\20200318'

#%% perform extraction
motion_file, dview = pipe.motion_correction(root, opts, dview, remove_f_order=True, remove_c_order=True)

root = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M32\20200322'

mmap_file, images = pipe.load_mmap(root)
cnm_params = cnm_params.change_params({'fnames': mmap_file[0]})
cnm = pipe.run_source_extraction(images, cnm_params, dview=dview)
pipe.save_cnmf(cnm, path=os.path.join(root, 'cnm_pre_selection.hdf5'), verbose=False, overwrite=True)
lcm = cm.local_correlations(images, swap_dim=False)
lcm[np.isnan(lcm)] = 0
cnm.estimates.Cn = lcm
pipe.save_cnmf(cnm, path=os.path.join(root, 'cnm_pre_selection.hdf5'), verbose=False, overwrite=True)
cnm.estimates.plot_contours(img=cnm.estimates.Cn, display_numbers=False)
plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches((10, 10))
plt.savefig(os.path.join(root, 'pre_sel_components.png'))
pipe.save_cnmf(cnm, path=os.path.join(root, 'cnm_pre_selection.hdf5'), verbose=False, overwrite=True)
plt.close()


#%% load pre-extracted data
cnm = pipe.load_cnmf(root, 'cnm_pre_selection.hdf5')
mmap_file, images = pipe.load_mmap(root)
#%% COMPONENT EVALUATION
# the components are evaluated in three ways:
#   a) the shape of each component must be correlated with the data
#   b) a minimum peak SNR is required over the length of a transient
#   c) each shape passes a CNN based classifier
min_SNR = 10  # signal to noise ratio for accepting a component (default 2)
SNR_lowest = 5.5
rval_thr = 0.95  # space correlation threshold for accepting a component (default 0.85)
cnn_thr = 0.99  # threshold for CNN based classifier (default 0.99)
cnn_lowest = 0.05  # neurons with cnn probability lower than this value are rejected (default 0.1)

cnm.params.set('quality', {'SNR_lowest': SNR_lowest,
                           'min_SNR': min_SNR,
                           'rval_thr': rval_thr,
                           'use_cnn': True,
                           'min_cnn_thr': cnn_thr,
                           'cnn_lowest': cnn_lowest})
cnm = pipe.run_evaluation(images, cnm, dview=dview)

cnm.estimates.plot_contours(img=cnm.estimates.Cn, idx=cnm.estimates.idx_components, display_numbers=True)

pipe.save_cnmf(cnm, path=os.path.join(root, 'cnm_pre_selection.hdf5'), verbose=False, overwrite=True)
cnm.estimates.select_components(use_object=True)
cnm.estimates.detrend_df_f(quantileMin=8, frames_window=int(len(cnm.estimates.C[0])/3))
cnm.params.data['dff_window'] = int(len(cnm.estimates.C[0])/3)
pipe.save_cnmf(cnm, path=os.path.join(root, 'cnm_results.hdf5'), overwrite=True, verbose=False)
cnm.estimates.plot_contours(img=cnm.estimates.Cn, display_numbers=False)
plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches((10, 10))
plt.savefig(os.path.join(root, 'components.png'))
plt.close()

pcf_params['root'] = root

pcf = pc.PlaceCellFinder(cnm, pcf_params)
# split traces into trials
pcf.split_traces_into_trials()
# align the frames to the VR position using merged behavioral data
pcf.save()
pcf.import_behavior_and_align_traces(remove_resting_frames=True)
pcf.params['remove_resting_frames'] = True
# create significant-transient-only traces
pcf.create_transient_only_traces()
pcf.save(overwrite=True)
# look for place cells
pcf.find_place_cells()
# save pcf object
if len(pcf.place_cells) > 0:
    pcf.plot_all_place_cells(save=False, show_neuron_id=True)
pcf.save(overwrite=True)

plt.savefig(os.path.join(root, 'eval_components.png'))
pipe.save_cnmf(cnm, root=root)
print(f'Finished with {root}!')

cnm.estimates.view_components(images, img=cnm.estimates.Cn,
                               idx=cnm.estimates.idx_components)
cnm.estimates.view_components(images, img=cnm.estimates.Cn,
                               idx=cnm.estimates.idx_components_bad)
cnm.estimates.select_components(use_object=True)

cnm.estimates.view_components(img=cnm.estimates.Cn)

#pipe.save_cnmf(cnm, root=root)
#%% Initialize PlaceCellFinder object

#cnm = pipe.load_cnmf(root)
params = {'root': root,                  # main directory of this session
          'trans_length': 0.5,           # minimum length in seconds of a significant transient
          'trans_thresh': 4,             # factor of sigma above which a transient is significant
          'bin_length': 5,               # length in cm VR distance of each bin in which to group the dF/F traces (has to be divisor of track_length
          'bin_window_avg': 3,           # sliding window of bins (left and right) for trace smoothing
          'bin_base': 0.25,              # fraction of lowest bins that are averaged for baseline calculation
          'place_thresh': 0.25,          # threshold of being considered for place fields, calculated
                                         # from difference between max and baseline dF/F
          'min_pf_size': 15,          # minimum size in cm for a place field (should be 15-20 cm)
          'fluo_infield': 7,             # factor above which the mean DF/F in the place field should lie compared to outside the field
          'trans_time': 0.2,             # fraction of the (unbinned!) signal while the mouse is located in
                                         # the place field that should consist of significant transients
          'track_length': 400,           # length in cm of the virtual reality corridor
          'split_size': 10}              # size in frames of bootstrapping segments

#%%
# r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M18\20191203a\N1',
# r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M18\20191203b\N1',
# r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M18\20191205\N1',
# r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M18\20191206\N1',

roots = [r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M18\20191207\N1',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M18\20191208\N1',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M25\20191127a\N1',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M25\20191206\N1',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M25\20191207\N1',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M25\20191208\N1']

count = 0
for root in roots:
    cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)
    mmap_file, images = pipe.load_mmap(root)
    cnm_params = cnm_params.change_params({'fnames': mmap_file[0]})
    if count%2 == 0:
        cnm_params = cnm_params.change_params({'nb': 3})
    else:
        cnm_params = cnm_params.change_params({'nb': 2})
    cnm = pipe.run_source_extraction(images, cnm_params, dview=dview)
    pipe.save_cnmf(cnm, path=os.path.join(root, 'cnm_pre_selection.hdf5'), verbose=False, overwrite=True)
    lcm = cm.local_correlations(images, swap_dim=False)
    lcm[np.isnan(lcm)] = 0
    cnm.estimates.Cn = lcm
    pipe.save_cnmf(cnm, path=os.path.join(root, 'cnm_pre_selection.hdf5'), verbose=False, overwrite=True)
    cnm.estimates.plot_contours(img=cnm.estimates.Cn, display_numbers=False)
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches((10, 10))
    plt.savefig(os.path.join(root, 'pre_sel_components.png'))
    plt.close()
    count += 1
    print('Done! \n \n \n \n')

#%% Performance evaluation



path = [r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M32',
        r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M33',
        r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M35',
        r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M36',
        r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M37',
        r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M38',
        r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M39',
        r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M40',
        r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M41']

path = [r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3']

data = performance.load_performance_data(roots=path, novel=False, norm_date=None)

performance.plot_all_mice_avg(data, rotate_labels=True)
performance.plot_all_mice_separately(data, rotate_labels=True)
performance.plot_single_mouse(data, 'M39')

def compare_trans_only(obj, idx):
    n_cols = 3
    n_rows = obj.params['n_trial']+1

    fig, ax = plt.subplots(n_rows, n_cols)

    data = obj.session[idx]
    data_trans = obj.session_trans[idx]

    y_min = min(np.hstack(data))
    y_max = max(np.hstack(data))

    for i in range(n_rows):
        if i < n_rows-1:
            ax[i, 0].plot(data[i])
            ax[i, 0].set_xticks([])
            ax[i, 2].set_ylim(y_min, y_max)
            ax[i, 1].plot(data_trans[i])
            ax[i, 1].set_xticks([])
            ax[i, 2].set_ylim(y_min, y_max)
            ax[i, 2].plot(obj.bin_activity[idx][i])
            ax[i, 2].set_xticks([])
            ax[i, 2].set_ylim(y_min, y_max)

        else:
            ax[i, 2].plot(obj.bin_avg_activity[idx])
            ax[i, 2].set_xticks([])

#%% avg compute test
path = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M39\20200318'
mmap_file, images = pipe.load_mmap(path)
avg = np.zeros(images.shape[1:])
tot_pix = images.shape[1]*images.shape[2]
count = 1
for row in range(images.shape[1]):
    for col in range(images.shape[2]):
        curr_pix = images[:, row, col]
        avg[row, col] = np.mean(curr_pix)
        pipe.progress(count, tot_pix, status=f'Computed {count}/{int(tot_pix)} pixels...')
        count += 1


