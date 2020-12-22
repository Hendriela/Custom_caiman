from caiman.source_extraction import cnmf
from standard_pipeline import place_cell_pipeline as pipe, behavior_import as behavior, \
    performance_check as performance
import div.file_manager as fm
import place_cell_class as pc
import caiman as cm
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
#import manual_neuron_selection_gui as selection_gui
import multisession_analysis.dimensionality_reduction as dr
import multisession_analysis.multisession_registration as msr
from scipy.stats import ttest_ind
import pandas as pd
import pickle
import multisession_analysis.batch_analysis as batch

"""
Complete pipeline for place cell analysis, from motion correction to place cell finding

Condition before starting:
Tiff and behavioral files (encoder, position, frames/licks) have to be in separate folders for each trial, and trial
folders have to be grouped in one folder per field of view. Several FOVs can be grouped in a folder per session.
"""
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)
#%% Manual cell tracking tool

# Which sessions should be aligned?
session_list = [r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M33\20200818',
                r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M33\20200819',
                r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M33\20200820',
                r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M33\20200821',
                r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M33\20200824',
                r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M33\20200826',
                r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M33\20200827',
                r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M33\20200830',
                r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M33\20200902',
                r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M33\20200905',
                r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M33\20200908',
                r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M33\20200911']

# This function loads the data of all sessions and stores it in the list "pcf_objects"
spatial, templates, dim, pcf_objects = msr.load_multisession_data(session_list)

# Which session should be the reference (place cells from this session will be tracked)
reference_session = '20200826'

# This function prepares the data for the tracking process
target_session_list, place_cell_indices, alignment_array, all_contours_list, all_shifts_list = msr.prepare_manual_alignment_data(
                                                                                                          pcf_objects, reference_session)

# If you started to align the place cells of this session, but saved the table incomplete and want to continue, you can
# load the table here and pick up where you left. Note that the file path should include the file name and extension
file_path = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\cell_alignments\pc_alignment_20200826.txt'
alignment_array = msr.load_alignment(file_path)

# This is the main function for the tracking. It creates the interactive plot and saves the results in the
# alignment_array, which has one place cell for each row, one session in each column, and each entry is the neuron ID
# that the reference cell has in each session.
alignment_array = msr.manual_place_cell_alignment(pcf_sessions=pcf_objects,
                                                  target_sessions=target_session_list,
                                                  cell_idx=place_cell_indices,
                                                  alignment=alignment_array,
                                                  all_contours=all_contours_list,
                                                  all_shifts=all_shifts_list,
                                                  ref_sess=reference_session,
                                                  dim=dim,
                                                  show_neuron_id=True)

# In case the correct cell is not displayed in the alignment plot, you can show the whole FOVs of the reference as well
# as the target session and look for the correct cell yourself. Use the indices displayed in the interactive graph to
# access the correct session from the list and the correct cell ID.
msr.show_whole_fov(reference_session=pcf_objects[5], target_session=pcf_objects[2], ref_cell_id=120)

# Save the alignment array under the provided directory as a csv table. Every row is one place cell from the reference
# session, every column is one session, and the entries are the IDs of each cell in the corresponding session.
file_directory = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\cell_alignments'
msr.save_alignment(file_directory, alignment_array, reference_session, pcf_objects)

count = 0
for idx, i in enumerate(pcf.params['frame_list']):
    count += i
    print(f'Trial {idx+1}, Frame {count-i} to {count}')

align_list = [r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\cell_alignments\pc_alignment_M33_20200818.txt',
              r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\cell_alignments\pc_alignment_M33_20200819.txt',
              r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\cell_alignments\pc_alignment_M33_20200820.txt',
              r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\cell_alignments\pc_alignment_M33_20200821.txt',
              r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\cell_alignments\pc_alignment_M33_20200824.txt']

alignments = msr.load_alignment(align_list)


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
rf = 50
# half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
stride_cnmf = 20  # amount of overlap between the patches in pixels (20)
K = 12  # number of components per patch (10)
gSig = [12, 12]  # expected half-size of neurons in pixels
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
          'trans_time': 0.15,             # fraction of the (unbinned!) signal while the mouse is located in
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
params = {'root': step[0],                  # main directory of this session
          'trans_length': 0.5,           # minimum length in seconds of a significant transient
          'trans_thresh': 4,             # factor of sigma above which a transient is significant
          'bin_length': 5,               # length in cm VR distance of each bin in which to group the dF/F traces (has to be divisor of track_length
          'bin_window_avg': 3,           # sliding window of bins (left and right) for trace smoothing
          'bin_base': 0.25,              # fraction of lowest bins that are averaged for baseline calculation
          'place_thresh': 0.25,          # threshold of being considered for place fields, calculated
                                         # from difference between max and baseline dF/F
          'min_pf_size': 15,          # minimum size in cm for a place field (should be 15-20 cm)
          'fluo_infield': 7,             # factor above which the mean DF/F in the place field should lie compared to outside the field
          'trans_time': 0.15,             # fraction of the (unbinned!) signal while the mouse is located in
                                         # the place field that should consist of significant transients
          'track_length': 400,           # length in cm of the virtual reality corridor
          'split_size': 10}              # size in frames of bootstrapping segments

#%%

# r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M18\20191203a\N1',
# r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M18\20191203b\N1',
# r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M18\20191205\N1',
# r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M18\20191206\N1',

roots = [r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M41\20200508',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M41\20200509',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M41\20200510',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M41\20200511']

for root in roots:

    cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)

    # dataset dependent parameters
    fr = 30  # imaging rate in frames per second
    decay_time = 0.4  # length of a typical transient in seconds (0.4)
    dxy = (1.66, 1.52)  # spatial resolution in x and y in (um per pixel) [(1.66, 1.52) for 1x, (0.83, 0.76) for 2x]

    # extraction parameters
    p = 1  # order of the autoregressive system
    gnb = 3  # number of global background components (3)
    merge_thr = 0.75  # merging threshold, max correlation allowed (0.86)
    rf = 25  # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
    stride_cnmf = 10  # amount of overlap between the patches in pixels (20)
    K = 23  # number of components per patch (10)
    gSig = [5, 5]  # expected half-size of neurons in pixels [X, Y] (has to be int, not float!)
    method_init = 'greedy_roi'  # initialization method (if analyzing dendritic data using 'sparse_nmf')
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

    cnm_params = cnmf.params.CNMFParams(params_dict=opts_dict)

    # Load memmap file
    mmap_file, images = pipe.load_mmap(root)
    cnm_params = cnm_params.change_params({'fnames': mmap_file})

    # # Run source extraction
    cnm = pipe.run_source_extraction(images, cnm_params, dview=dview)
    pipe.save_cnmf(cnm, path=os.path.join(root, 'cnm_pre_selection.hdf5'), verbose=False, overwrite=True)

    # Load local correlation image (should have been created during motion correction)
    try:
        cnm.estimates.Cn = io.imread(root + r'\local_correlation_image.tif')
    except FileNotFoundError:
        pipe.save_local_correlation(images, root)
        cnm.estimates.Cn = io.imread(root + r'\local_correlation_image.tif')
    pipe.save_cnmf(cnm, path=os.path.join(root, 'cnm_pre_selection.hdf5'), verbose=False, overwrite=True)

    # Plot and save contours of all components
    cnm.estimates.plot_contours(img=cnm.estimates.Cn, display_numbers=False)
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches((10, 10))
    plt.savefig(os.path.join(root, 'pre_sel_components.png'))
    pipe.save_cnmf(cnm, path=os.path.join(root, 'cnm_pre_selection.hdf5'), verbose=False, overwrite=True)

#%% Performance evaluation

path = [r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M33',
        r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M38',
        r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M39',
        r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M41']

path = [r'W:\Neurophysiology-Storage1\Wahl\Jithin\VR data\M2',
        r'W:\Neurophysiology-Storage1\Wahl\Jithin\VR data\M7']

path = [r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3']

stroke = ['M32', 'M40', 'M41']
control = ['M33', 'M38', 'M39']

data = performance.load_performance_data(roots=path, norm_date='20200513', stroke=stroke)
data = data[data['session_date'] != '20200826']
data = performance.normalize_performance(data, ('20200818', '20200824'))


performance.plot_all_mice_avg(data, rotate_labels=False, field='licking_binned_norm',
                              session_range=(60, 73), scale=2)

performance.plot_all_mice_separately(data, field='licking_binned', x_axis='session_date', rotate_labels=True, scale=1.75, vlines=[8.5])
sns.set_context('talk')
axis = performance.plot_single_mouse(data, 'M41', field='licking', session_range=(26, 47))
axis = performance.plot_single_mouse(data, 'M32', session_range=(10, 15), scale=2, ax=axis)

# Filter data to exclude sessions and mice
filter_data = data[data['session_date'] != '20200826']
filter_data = data[data['sess_norm'] >= -8]
filter_data = filter_data[filter_data['sess_norm'] <= 31]
filter_data = filter_data[filter_data['mouse'] != 'M40']
filter_data = filter_data[filter_data['mouse'] != 'M35']

# Exclude days with changed VR from Jithins endothelin
m2_data = data[(data['mouse'] == 'M2') & (data['session_date'] != '20200728')]
m2_data = m2_data[(m2_data['mouse'] == 'M2') & (m2_data['session_date'] != '20200730')]
m2_data = m2_data[(m2_data['mouse'] == 'M2') & (m2_data['session_date'] != '20200731')]
m2_data = m2_data[(m2_data['mouse'] == 'M2') & (m2_data['session_date'] != '20200804')]
m7_data = data[(data['mouse'] == 'M7') & (data['session_date'] != '20200728')]
filter_data = pd.concat((m2_data, m7_data))

performance.plot_all_mice_separately(filter_data, field='licking_binned', x_axis='session_date',
                                     rotate_labels=True, scale=1.75, vlines=[8.5])


# Get performance baselines
df = performance.normalize_performance(filter_data, ('20200507', '20200511'))

# Export in prism format
batch.exp_to_prism_mouse_avg(filter_data, 'licking_binned_norm')

plt.figure()
filter_data = data[data['session_date'] != '20200826']
filter_data = data[data['sess_norm'] >= -4]
filter_data = filter_data[filter_data['sess_norm'] <= 31]
sns.set()
sns.lineplot(x='sess_norm', y='licking_binned', hue='group', data=filter_data)
plt.axvline(-1, color='r')
plt.axvline(7, color='r')
plt.axvline(69, color='r')
plt.axvline(21, color='r')

# Plot simple data
simple_data = pd.read_pickle(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing'
                             r'\simple_data_first_stroke.pickle')

plt.figure()
sns.lineplot(x='sess_norm', y='pvc_slope_norm', data=simple_data, hue='group', palette=['red', 'black'])
plt.axvline(5, color='red')
plt.ylabel('normalized max PVC slope')
plt.xlabel('days')

from scipy.stats import ttest_ind
stroke_lick = np.array(data[(data['group'] == 'stroke') & (data['session_date'] == '20200507')]['licking'])
control_lick = np.array(data[(data['group'] == 'control') & (data['session_date'] == '20200507')]['licking'])
stat, p = ttest_ind(stroke_lick, control_lick)
print(p)

avg_stroke_lick = []
avg_control_lick = []
for mouse in stroke:
    avg_stroke_lick.append(np.mean(np.array(data[(data['group'] == 'stroke') &
                                                 (data['session_date'] == '20200511') &
                                                 (data['mouse'] == mouse)]['licking'])))
for mouse in control:
    avg_control_lick.append(np.mean(np.array(data[(data['group'] == 'control') &
                                                  (data['session_date'] == '20200513') &
                                                  (data['mouse'] == mouse)]['licking'])))

stat, p = ttest_ind(avg_stroke_lick, avg_control_lick)


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


#%%
from spike_prediction.spike_prediction import predict_spikes
roots = [r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M39\20200320',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M39\20200321',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M39\20200322',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M39\20200323',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M40\20200318',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M40\20200319',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M40\20200320',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M40\20200321',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M40\20200322']

for root in roots:

    # Set parameters
    pcf_params = {'root': root,  # main directory of this session
                  'trans_length': 0.5,  # minimum length in seconds of a significant transient
                  'trans_thresh': 4,  # factor of sigma above which a transient is significant
                  'bin_length': 5,
                  # length in cm VR distance in which to bin dF/F trace (must be divisor of track_length)
                  'bin_window_avg': 3,  # sliding window of bins (left and right) for trace smoothing
                  'bin_base': 0.25,  # fraction of lowest bins that are averaged for baseline calculation
                  'place_thresh': 0.25,  # threshold of being considered for place fields, calculated
                  #     from difference between max and baseline dF/F
                  'min_pf_size': 15,  # minimum size in cm for a place field (should be 15-20 cm)
                  'fluo_infield': 7,
                  # factor above which the mean DF/F in the place field should lie vs. outside the field
                  'trans_time': 0.2,  # fraction of the (unbinned!) signal while the mouse is located in
                  # the place field that should consist of significant transients
                  'track_length': 400,  # length in cm of the virtual reality corridor
                  'split_size': 50}  # size in frames of bootstrapping segments

    # Load CNM object
    cnm = pipe.load_cnmf(root)

    # Initialize PCF object with the raw data (CNM object) and the parameter dict
    pcf = pc.PlaceCellFinder(cnm, pcf_params)
    old_pcf = pipe.load_pcf(root, 'pcf_results_save.pickle')

    # If necessary, perform Peters spike prediction
    pcf.cnmf.estimates.spikes = old_pcf.cnmf.estimates.spikes
    pcf.cnmf.estimates.spikes = predict_spikes(pcf.cnmf.estimates.F_dff)

    # split traces into trials
    pcf.split_traces_into_trials()

    # Import behavior and align traces to it, while removing resting frames
    pcf.import_behavior_and_align_traces()
    pcf.params['resting_removed'] = True
    pcf.bin_activity_to_vr(remove_resting=pcf.params['resting_removed'])

    # # create significant-transient-only traces
    pcf.create_transient_only_traces()

    pcf.save(overwrite=True)

#%%
import caiman as cm
import matplotlib.pyplot as plt
import os
import numpy as np
import standard_pipeline.place_cell_pipeline as pipe

roots = [r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M41\20200507',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M41\20200508',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M41\20200509',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M41\20200510']

for root in roots:
    # Start cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)

    # Load CNM object with pre-selected components
    cnm = pipe.load_cnmf(root, 'cnm_pre_selection.hdf5')

    # Load movie
    mmap_file, images = pipe.load_mmap(root)
    # evaluation parameters
    min_SNR = 8  # signal to noise ratio for accepting a component (default 2)
    SNR_lowest = 3.7
    rval_thr = 0.85  # space correlation threshold for accepting a component (default 0.85)
    rval_lowest = -1
    cnn_thr = 0.83  # threshold for CNN based classifier (default 0.99)
    cnn_lowest = 0.18  # neurons with cnn probability lower than this value are rejected (default 0.1)

    cnm.params.set('quality', {'SNR_lowest': SNR_lowest,
                               'min_SNR': min_SNR,
                               'rval_thr': rval_thr,
                               'rval_lowest': rval_lowest,
                               'use_cnn': True,
                               'min_cnn_thr': cnn_thr,
                               'cnn_lowest': cnn_lowest})
    cnm = pipe.run_evaluation(images, cnm, dview=dview)

    # Save the CNM object once before before deleting the data of rejected components
    pipe.save_cnmf(cnm, path=os.path.join(root, 'cnm_pre_selection.hdf5'), verbose=False, overwrite=True)

    # Select components, which keeps the data of accepted components and deletes the data of rejected ones
    cnm.estimates.select_components(use_object=True)

    cnm.params.data['dff_window'] = 2000
    cnm.estimates.detrend_df_f(quantileMin=8, frames_window=cnm.params.data['dff_window'])

    # Save complete CNMF results
    pipe.save_cnmf(cnm, path=os.path.join(root, 'cnm_results.hdf5'), overwrite=False, verbose=False)

    # Plot contours of all accepted components
    cnm.estimates.plot_contours(img=cnm.estimates.Cn, display_numbers=False)
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches((10, 10))
    plt.savefig(os.path.join(root, 'components.png'))
    plt.close()

    cm.stop_server(dview=dview)

#%% cross-session alignment groundtruth export Anna

from caiman.utils import visualization

paths = [r'20191122a',r'20191125',r'20191126b', '20191127a',r'20191204',r'20191205',r'20191206',
         r'20191207',r'20191208',r'20191219']
target = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\batch_analysis\alignment_session_data'
files = os.listdir(target)

for i, sess in enumerate(paths):
    pcf = pipe.load_pcf(r'E:\Batch2\M19\{}\N2'.format(sess))
    data = np.load(os.path.join(target, files[i+1]), allow_pickle=True)
    del data['contours']
    out = visualization.plot_contours(pcf.cnmf.estimates.A, pcf.cnmf.estimates.Cn, display_numbers=False, colors='r', verbose=False)
    plt.close()
    data['template'] = pcf.cnmf.estimates.Cn
    data['CoM'] = np.vstack([x['CoM'] for x in out])
    np.save(os.path.join(target, os.path.splitext(files[i+1])[0]), data, allow_pickle=True)


#%% Dimensionality reduction

target = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\PCA'

mice = [r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M32',
        r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M33',
        r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M38',
        r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M39',
        r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M40',
        r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M41']

sessions = ['20200507', '20200508', '20200509', '20200510', '20200511', '20200513', '20200514',
            '20200515', '20200516', '20200517', '20200518', '20200519']


# pc_dist_diff = np.zeros((3, len(paths)))
df_list = []
data_list = []
session_list = []

for mouse_dir in mice:
    mouse = mouse_dir.split(os.path.sep)[-1]
    curr_target = os.path.join(target, mouse)
    for sess in sessions:
        pcf = pipe.load_pcf(os.path.join(mouse_dir, sess))
        data, labels, model = dr.perform_pca(pcf)
        scores = model.transform(data)
        weights = model.components_
        variance_explained = model.explained_variance_ratio_

        session = pcf.params['session']

        # Plot (cumulative) variance explained
        cutoff = dr.plot_variance_explained(variance_explained, return_cutoff=True)
        plt.title(f'Explained variance {mouse}_{session}')
        fname = f'{mouse}_{session}_explained_variance.png'
        plt.savefig(os.path.join(curr_target, fname))
        plt.close()

        # Plot weight profiles of first X components
        dr.plot_weights(weights, n_comps=12, params=pcf.params, var_exp=variance_explained)
        fig = plt.gcf()
        fig.suptitle(f'Weight profiles {mouse}_{session}')
        plt.subplots_adjust(top=0.92)
        fname = f'{mouse}_{session}_weights.png'
        plt.savefig(os.path.join(curr_target, fname))
        plt.close()

        # Plot first two principal components with histograms
        fig = dr.plot_pc_with_hist(scores, weights, labels, pcf.params)
        fig.suptitle(f'First two principal components of {mouse}_{session}')
        fname = f'{mouse}_{session}_principal_components.png'
        fig.savefig(os.path.join(curr_target, fname))
        plt.close()

        # Calculate difference in mean and median of place cells vs non-pcs across PC1
        pc1 = scores[:, 0]
        mean_diff = abs(np.mean(pc1[labels]) - np.mean(pc1[~labels]))
        # pc_dist_diff[1, idx] = np.median(pc1[labels]) - np.mean(pc1[~labels])
        # t, p = ttest_ind(pc1[labels], pc1[~labels], equal_var=False)
        # pc_dist_diff[2, idx] = p

        # Put data together in one dataframe
        df_list.append(pd.DataFrame({'mouse': mouse, 'session': session, 'pca_model': model,
                                     '95%_variance': [cutoff], 'place_cell_diff': [mean_diff]}))
        data_list.append(data)
        session_list.append(os.path.join(mouse_dir, sess))

df = pd.concat(df_list)

# Normalize data
df['session'] = df['session'].astype(int)
mouse_list = ['M32', 'M33', 'M38', 'M39', 'M40', 'M41']
stroke = ['M32', 'M40', 'M41']
df['norm_var'] = 0.
df['norm_pc_diff'] = 0.
for mouse in mouse_list:
    curr_df = df.loc[df['mouse'] == mouse]
    baseline_var = np.nanmean(curr_df.loc[curr_df['session'] < 20200513, '95%_variance'])
    df.loc[df['mouse'] == mouse, 'norm_var'] = curr_df['95%_variance'] / baseline_var
    baseline_diff = np.nanmean(curr_df.loc[curr_df['session'] < 20200513, 'place_cell_diff'])
    df.loc[df['mouse'] == mouse, 'norm_pc_diff'] = curr_df['place_cell_diff'] / baseline_diff

all_sess = sorted(df['session'].unique())
count = 0
df['sess_id'] = -1
for session in all_sess:
    df.loc[df['session'] == session, 'sess_id'] = count
    count += 1
df['group'] = 'unknown'
if stroke is not None:
    df.loc[df.mouse.isin(stroke), 'group'] = 'stroke'
    df.loc[~df.mouse.isin(stroke), 'group'] = 'control'

data_key = [data_list, session_list]
with open(r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\PCA\pca_data.pickle", "wb") as fp:   #Pickling
    pickle.dump(data_key, fp)
df.to_pickle(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\PCA\pca_results.pickle')

# Plot groups together
sns.lineplot(x='sess_id', y='place_cell_diff', hue='group', data=df)
plt.axvline(4.5, color='r')
plt.ylabel('Difference PC vs non-PC distributions')
# plot mice average
grid = sns.FacetGrid(df, col='mouse', col_wrap=3, height=3, aspect=2)
grid.map(sns.lineplot, 'sess_id', '')
grid.set_axis_labels('session', 'Difference PC vs non-PC distributions')
for ax in grid.axes.ravel():
    ax.axvline(4.5, color='r')


######## t-SNE

target = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\t-SNE'

df_list = []
data_list = []
session_list = []

for mouse_dir in mice:
    mouse = mouse_dir.split(os.path.sep)[-1]
    curr_results = []
    curr_labels = []
    for sess in sessions:
        pcf = pipe.load_pcf(os.path.join(mouse_dir, sess))
        data, labels, tsne_mod, embed = dr.perform_tsne(pcf, 50)
        df_list.append(pd.DataFrame({'mouse': mouse, 'session': sess, 'comp_1': embed[:, 0], 'comp_2': embed[:, 1],
                                     'labels': labels}))
        curr_results.append(embed)
        curr_labels.append(labels)

    fig, ax = plt.subplots(3, 4, figsize=(12,10))
    count = 0
    for row in range(3):
        for col in range(4):
            ax[row, col].scatter(x=curr_results[count][:, 0], y=curr_results[count][:, 1], c=curr_labels[count], s=15)
            ax[row, col].set_title(f'Session {sessions[count]}')
            ax[row, col].tick_params(direction='in', labelbottom=False, labelleft=False)

            count += 1

    plt.tight_layout()
    fig.suptitle(f't-SNE Mouse {mouse} (perplexity 50)')
    plt.subplots_adjust(top=0.92)
    fname = target+'\\'+f'{mouse}_perplexity_50.png'
    plt.savefig(fname)

df = pd.concat(df_list)
df.to_pickle(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\t-SNE\tsne_results.pickle')

# run collective t-SNE on all sessions of one mouse combined
mouse_dir = mice[-1]
mouse = 'M41'
df_list = []
for sess in sessions:
    pcf = pipe.load_pcf(os.path.join(mouse_dir, sess))
    # data, labels, tsne_mod, embed = dr.perform_tsne(pcf, 50)
    df = pd.DataFrame(pcf.bin_avg_activity)
    pc_idx = [x[0] for x in pcf.place_cells]
    labels = np.zeros(len(pcf.bin_avg_activity))
    labels[pc_idx] = 1
    df['place_cells'] = labels
    df['mouse'] = mouse
    df['session'] = sess
    df_list.append(df)
df = pd.concat(df_list)

# Standardize data
for i in range(80):
    df[i] = (df[i] - np.mean(df[i])) / np.std(df[i])

# Perform t-SNE
tsne_mod = TSNE(n_components=2, perplexity=50, n_iter=1000)
embed = tsne_mod.fit_transform(df.iloc[:, range(80)])
df['comp1'] = embed[:, 0]
df['comp2'] = embed[:, 1]
df['prestroke'] = df['sess_id'] < 5

sns.scatterplot(x="comp1", y="comp2", hue="place_cells", data=df, legend="full")

# Perform t-SNE with different perplexities
fig, ax = plt.subplots(2, 3)
perplexities = [5, 30, 50, 75, 100, 500]
count = 0
for row in range(2):
    for col in range(3):
        pca_mod = PCA(n_components=50)
        pca_results = pca_mod.fit_transform(df.iloc[:, range(80)])
        tsne_mod = TSNE(n_components=2, perplexity=perplexities[count], n_iter=1000)
        embed = tsne_mod.fit_transform(pca_results)
        ax[row, col].scatter(x=embed[:, 0], y=embed[:, 1], c=df['place_cells'], s=10)
        ax[row, col].set_xlabel('Component 1')
        ax[row, col].set_ylabel('Component 2')
        ax[row, col].set_title(f'Perplexity {perplexities[count]}')
        count += 1

# load t-SNE results
tsne = pd.read_pickle(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\t-SNE\tsne_results.pickle')


#%% Simple data plotting

df = pd.read_pickle(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\\Batch3\batch_processing\simple_data.pickle')


norm_fields = ['n_cells', 'n_place_cells', 'ratio', 'mean_spikerate', 'median_spikerate', 'pvc_slope',
               'min_pvc', 'sec_peak_ratio']
norm_range = (20200507, 20200511)
for field in norm_fields:
    # Find indices of correct rows (pre-stroke sessions for the specific animal)
    df[field + '_norm'] = -1.0
    for mouse in df.mouse.unique():
        norm_factor = df.loc[(df.mouse == mouse) &
                             ((df.session >= norm_range[0]) & (df.session <= norm_range[1])), field].mean()
        df.loc[df.mouse == mouse, field + '_norm'] = df.loc[df.mouse == mouse, field] / norm_factor

strokes = [11.5, 18.5, 23.5, 26.5]
target = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\laser_stroke_plots'
for field in norm_fields:
    batch.plot_simple_data_single_mice(data, field=field, stroke_sess=strokes)
    plt.savefig(os.path.join(target, f'{field}_sep_notnorm.png'))
    plt.close()
    batch.plot_simple_data_group_avg(data, field=field, stroke_sess=strokes)
    plt.savefig(os.path.join(target, f'{field}_avg_notnorm.png'))
    plt.close()

# Filter for microsphere data
df = df.loc[(df['mouse'] != 'M35')]
data = df.loc[(df['session'] >= 20200507) & (df['session'] <= 20200611)]

filter_data['group'] = 'lesion'
# Saves plots of all normed data
target = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\microsphere_plots'
for field in norm_fields:
    batch.plot_simple_data_single_mice(filter_data, field=field, stroke_sess=[45.5])
    plt.savefig(os.path.join(target, f'{field}_sep_notnorm.png'))
    plt.close()
    batch.plot_simple_data_group_avg(filter_data, field=field, stroke_sess=[45.5])
    plt.savefig(os.path.join(target, f'{field}_avg_notnorm.png'))
    plt.close()


# Plot PVC curves
mice = filter_data['mouse'].unique()
sessions = np.sort(filter_data['session'].unique())
fig, axes = plt.subplots(4, 12, sharex='all', sharey='all', figsize=(18,12))

for i, mouse in enumerate(mice):
    for j, session in enumerate(sessions):
        curve = filter_data.loc[(filter_data['mouse'] == mouse) & (filter_data['session'] ==session), 'pvc_curve']
        if len(curve) > 0:
            axes[i, j].plot(np.linspace(0,150,31), curve.iloc[0])


for ax, col in zip(axes[0], sessions):
    ax.set_title(col)

for ax, row in zip(axes[:,0], mice):
    ax.set_ylabel(row, rotation=0, size='large')

plt.tight_layout()

#%% Plot comparison graphs between different trace datasets
fr = 30 # frame rate, set to 1 to plot traces against frame counts
cell = 49
fig, ax = plt.subplots(2, 3)
ax[0,2].plot(np.arange(len(cnm.estimates.F_dff[cell]))/fr, cnm.estimates.F_dff[cell])
ax[0,2].set_title('estimates.F_dff')
ax[0,2].axvline(5000/fr, color='r')
ax[0,2].axvline(5800/fr, color='r')
ax[0,1].plot(np.arange(len(cnm.estimates.F_dff[cell]))/fr, cnm.estimates.C[cell])
ax[0,1].set_title('estimates.C (inferred trace)')
ax[0,1].axvline(5000/fr, color='r')
ax[0,1].axvline(5800/fr, color='r')
ax[0,0].plot(np.arange(len(cnm.estimates.F_dff[cell]))/fr, cnm.estimates.C[cell]+cnm.estimates.R[cell])
ax[0,0].set_title('estimates.C + residuals ("filtered raw trace")')
ax[0,0].axvline(5000/fr, color='r')
ax[0,0].axvline(5800/fr, color='r')
ax[1,2].plot(np.arange(5000,5800)/fr, cnm.estimates.F_dff[cell, 5000:5800])
# ax[1,0].set_title('estimates.F_dff zoom')
ax[1,1].plot(np.arange(5000,5800)/fr, cnm.estimates.C[cell, 5000:5800])
# ax[1,1].set_title('estimates.C zoom')
ax[1,0].plot(np.arange(5000,5800)/fr, cnm.estimates.C[cell, 5000:5800]+cnm.estimates.R[cell, 5000:5800])
# ax[1,2].set_title('estimates.S zoom')
ax[1,0].set_xlabel('frames')
ax[1,1].set_xlabel('frames')
ax[1,2].set_xlabel('frames')

# Plotting background components
fig, ax = plt.subplots(2, 2)
ax[0,0].imshow(np.reshape(pcf.cnmf.estimates.b[:,0],pcf.cnmf.dims, order='F'))
ax[0,1].imshow(np.reshape(pcf.cnmf.estimates.b[:,1],pcf.cnmf.dims, order='F'))
ax[1,0].plot(np.arange(len(pcf.cnmf.estimates.f[0]))/fr, pcf.cnmf.estimates.f[0])
ax[1,1].plot(np.arange(len(pcf.cnmf.estimates.f[1]))/fr, pcf.cnmf.estimates.f[1])
ax[0,0].set_title('Background component 1')
ax[0,1].set_title('Background component 2')
ax[1,0].set_xlabel('time [s]')
ax[1,1].set_xlabel('time [s]')

# Zoomed-in temporal components
fig, ax = plt.subplots(2, 1)
ax[0].plot(np.arange(14700, 15300)/fr, pcf.cnmf.estimates.f[0, 14700:15300])
ax[1].plot(np.arange(14700, 15300)/fr, pcf.cnmf.estimates.f[1, 14700:15300])
ax[0].set_title('Zoom background comp 1')
ax[1].set_title('Zoom Background comp 2')
ax[0].set_xlabel('time [s]')
ax[1].set_xlabel('time [s]')

# Residuals and (reconstructed) raw data
fr = 30 # frame rate, set to 1 to plot traces against frame counts
cell = 49
fig, ax = plt.subplots(2, 3)
ax[0,0].plot(np.arange(len(pcf.cnmf.estimates.F_dff[cell]))/fr, pcf.cnmf.estimates.C[cell])
ax[0,0].set_title('denoised (inferred) temporal trace C')
ax[0,0].axvline(5000/fr, color='r')
ax[0,0].axvline(5800/fr, color='r')
ax[0,1].plot(np.arange(len(pcf.cnmf.estimates.F_dff[cell]))/fr, pcf.cnmf.estimates.R[cell])
ax[0,1].set_title('residual temporal trace R')
ax[0,1].axvline(5000/fr, color='r')
ax[0,1].axvline(5800/fr, color='r')
ax[0,2].plot(np.arange(len(pcf.cnmf.estimates.F_dff[cell]))/fr, pcf.cnmf.estimates.C[cell]+pcf.cnmf.estimates.R[cell])
ax[0,2].set_title('"filtered raw trace" (C + R)')
ax[0,2].axvline(5000/fr, color='r')
ax[0,2].axvline(5800/fr, color='r')
ax[1,0].plot(np.arange(5000,5800)/fr, pcf.cnmf.estimates.C[cell, 5000:5800])
# ax[1,0].set_title('estimates.F_dff zoom')
ax[1,1].plot(np.arange(5000,5800)/fr, pcf.cnmf.estimates.R[cell, 5000:5800])
# ax[1,1].set_title('estimates.C zoom')
ax[1,2].plot(np.arange(5000,5800)/fr, pcf.cnmf.estimates.C[cell, 5000:5800]+pcf.cnmf.estimates.R[cell, 5000:5800])
# ax[1,2].set_title('estimates.S zoom')
ax[1,0].set_xlabel('time [s]')
ax[1,1].set_xlabel('time [s]')
ax[1,2].set_xlabel('time [s]')


big_string = ''
for substring in my_list:
    big_string = big_string + ' ' + substring

A = pcf.cnmf.estimates.A
F =  pcf.cnmf.estimates.C +  pcf.cnmf.estimates.YrA
b = pcf.cnmf.estimates.b
f= pcf.cnmf.estimates.f
B = A.T.dot(b).dot(f)
import scipy.ndimage as nd
Df = nd.percentile_filter(B, 10, (1000,1))
plt.figure(); plt.plot(B[49]+pcf.cnmf.estimates.C[49]+pcf.cnmf.estimates.R[49])

#%% Flag auto as False, how does dFF look
import caiman.source_extraction.cnmf.utilities as ut
flag_dff = ut.detrend_df_f(A, b, pcf.cnmf.estimates.C, f, YrA=pcf.cnmf.estimates.YrA, quantileMin=8,
                           frames_window=500, flag_auto=False, use_fast=False, detrend_only=False)
slow_dff = ut.extract_DF_F(Yr, A, C, bl, quantileMin=8, frames_window=200, block_size=400, dview=None)

fig, ax = plt.subplots(2)
ax[0].plot(pcf.cnmf.estimates.F_dff[49])
ax[1].plot(flag_dff[49])

plt.figure()
plt.plot(flag_dff[49], label='auto=False')
plt.plot(pcf.cnmf.estimates.F_dff[49], label='auto=True')
plt.legend()

#%% Test with mean intensity and local correlation FOV shift for manual alignment tool
import tifffile as tif
from skimage.feature import register_translation
from scipy.ndimage import zoom


def piecewise_fov_shift(ref_img, tar_img, n_patch=8):
    """
    Calculates FOV-shift map between a reference and a target image. Images are split in n_patch X n_patch patches, and
    shift is calculated for each patch separately with phase correlation. The resulting shift map is scaled up and
    missing values interpolated to ref_img size to get an estimated shift value for each pixel.
    :param ref_img: np.array, reference image
    :param tar_img: np.array, target image to which FOV shift is calculated. Has to be same dimensions as ref_img
    :param n_patch: int, root number of patches the FOV should be subdivided into for piecewise phase correlation
    :return: two np.arrays containing estimated shifts per pixel (upscaled x_shift_map, upscaled y_shift_map)
    """
    img_dim = ref_img.shape
    patch_size = int(img_dim[0]/n_patch)

    shift_map_x = np.zeros((n_patch, n_patch))
    shift_map_y = np.zeros((n_patch, n_patch))
    for row in range(n_patch):
        for col in range(n_patch):
            curr_ref_patch = ref_img[row*patch_size:row*patch_size+patch_size, col*patch_size:col*patch_size+patch_size]
            curr_tar_patch = tar_img[row*patch_size:row*patch_size+patch_size, col*patch_size:col*patch_size+patch_size]
            patch_shift = register_translation(curr_ref_patch, curr_tar_patch, upsample_factor=100, return_error=False)
            shift_map_x[row, col] = patch_shift[0]
            shift_map_y[row, col] = patch_shift[1]
    shift_map_x_big = zoom(shift_map_x, patch_size, order=3)
    shift_map_y_big = zoom(shift_map_y, patch_size, order=3)
    return shift_map_x_big, shift_map_y_big

mean_int1 = tif.imread(r'W:\Neurophysiology-Storage1\Wahl\Jithin\Imaging\Batch 3\M31\Pre_Stroke\Session 1\Frontal\mean_intensity_image.tif')
mean_int2 = tif.imread(r'W:\Neurophysiology-Storage1\Wahl\Jithin\Imaging\Batch 3\M31\Pre_Stroke\Session 2\Frontal\mean_intensity_image.tif')
local_int1 = tif.imread(r'W:\Neurophysiology-Storage1\Wahl\Jithin\Imaging\Batch 3\M31\Pre_Stroke\Session 1\Frontal\local_correlation_image.tif')
local_int2 = tif.imread(r'W:\Neurophysiology-Storage1\Wahl\Jithin\Imaging\Batch 3\M31\Pre_Stroke\Session 2\Frontal\local_correlation_image.tif')

fig, ax = plt.subplots(2,3, sharex='row', sharey='row')
ax[0,0].imshow(mean_int1)
ax[0,1].imshow(mean_int2)
shiftx, shifty = piecewise_fov_shift(mean_int1, mean_int2)
im = ax[0,2].imshow(np.abs(shiftx)+np.abs(shifty))
fig.colorbar(im, ax=ax[0,2])

ax[1,0].imshow(local_int1)
ax[1,1].imshow(local_int2)
shiftx, shifty = piecewise_fov_shift(local_int1, local_int2)
im = ax[1,2].imshow(np.abs(shiftx)+np.abs(shifty))
fig.colorbar(im, ax=ax[1,2])

with ScanImageTiffReader(r'W:\Neurophysiology-Storage1\Wahl\Jithin\Imaging\Batch 3\M31\Pre_Stroke\Session 1\Frontal\mean_intensty_image.tif') as tif:
    print('yes')


