from caiman.source_extraction import cnmf
import place_cell_pipeline as pipe
import behavior_import as behavior
import place_cell_class as pc
import caiman as cm
import matplotlib.pyplot as plt
import os
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


#%% Set working directory

root = pipe.set_file_paths()

#%% Perform motion correction


for root in roots:
    if root == r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M22\20191129a':
        opts.change_params({'dxy': (0.83, 0.76)})
    motion_file, dview = pipe.motion_correction(root, opts, dview, remove_f_order=True, remove_c_order=True)

# save C-order files
mmap_files = [[r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M22\20191121b\N1\1\file_00003_els__d1_512_d2_512_d3_1_order_F_frames_3883_.mmap',
              r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M22\20191121b\N1\2\file_00004_els__d1_512_d2_512_d3_1_order_F_frames_2024_.mmap',
              r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M22\20191121b\N1\3\file_00005_els__d1_512_d2_512_d3_1_order_F_frames_1769_.mmap',
              r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M22\20191121b\N1\4\file_00006_els__d1_512_d2_512_d3_1_order_F_frames_2844_.mmap',
              r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M22\20191121b\N1\5\file_00007_els__d1_512_d2_512_d3_1_order_F_frames_2970_.mmap',
              r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M22\20191121b\N1\6\file_00008_els__d1_512_d2_512_d3_1_order_F_frames_3195_.mmap',
              r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M22\20191121b\N1\7\file_00009_els__d1_512_d2_512_d3_1_order_F_frames_6646_.mmap',
              r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M22\20191121b\N1\8\file_00010_els__d1_512_d2_512_d3_1_order_F_frames_3033_.mmap',
              r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M22\20191121b\N1\9\file_00011_els__d1_512_d2_512_d3_1_order_F_frames_2641_.mmap',
              r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M22\20191121b\N1\10\file_00012_els__d1_512_d2_512_d3_1_order_F_frames_3796_.mmap',
              r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M22\20191121b\N1\11\file_00013_els__d1_512_d2_512_d3_1_order_F_frames_2862_.mmap',
              r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M22\20191121b\N1\12\file_00014_els__d1_512_d2_512_d3_1_order_F_frames_4811_.mmap',
              r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M22\20191121b\N1\13\file_00015_els__d1_512_d2_512_d3_1_order_F_frames_3018_.mmap',
              r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M22\20191121b\N1\14\file_00016_els__d1_512_d2_512_d3_1_order_F_frames_1696_.mmap',
              r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M22\20191121b\N1\15\file_00017_els__d1_512_d2_512_d3_1_order_F_frames_1923_.mmap',
              r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M22\20191121b\N1\16\file_00018_els__d1_512_d2_512_d3_1_order_F_frames_2338_.mmap',
              r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M22\20191121b\N1\17\file_00019_els__d1_512_d2_512_d3_1_order_F_frames_1693_.mmap',
              r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M22\20191121b\N1\18\file_00020_els__d1_512_d2_512_d3_1_order_F_frames_1662_.mmap',
              r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M22\20191121b\N1\19\file_00021_els__d1_512_d2_512_d3_1_order_F_frames_2575_.mmap'],
              [r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M25\20191120\N1\1\file_00001_els__d1_512_d2_512_d3_1_order_F_frames_9889_.mmap',
               r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M25\20191120\N1\2\file_00002_els__d1_512_d2_512_d3_1_order_F_frames_1998_.mmap',
               r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M25\20191120\N1\3\file_00003_els__d1_512_d2_512_d3_1_order_F_frames_8622_.mmap',
               r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M25\20191120\N1\4\file_00004_els__d1_512_d2_512_d3_1_order_F_frames_1950_.mmap',
               r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M25\20191120\N1\5\file_00005_els__d1_512_d2_512_d3_1_order_F_frames_7251_.mmap',
               r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M25\20191120\N1\6\file_00006_els__d1_512_d2_512_d3_1_order_F_frames_4729_.mmap',
               r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M25\20191120\N1\7\file_00007_els__d1_512_d2_512_d3_1_order_F_frames_532_.mmap',
               r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M25\20191120\N1\8\file_00008_els__d1_512_d2_512_d3_1_order_F_frames_310_.mmap',
               r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M25\20191120\N1\9\file_00009_els__d1_512_d2_512_d3_1_order_F_frames_319_.mmap',
               r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M25\20191120\N1\10\file_00010_els__d1_512_d2_512_d3_1_order_F_frames_397_.mmap']]


#%% Align behavioral data
#behavior.align_files(folder_list, performance_check=True)
for root in roots:
    behavior.align_behavior(root, performance_check=False, verbose=False)

# evaluate behavior
mouse_list = []
#%% CaImAn source extraction
mmap_file, images = pipe.load_mmap(root)

#%%
p = 1  # order of the autoregressive system
gnb = 3  # number of global background components (3)
merge_thr = 0.70  # merging threshold, max correlation allowed (0.86)
rf = 25
# half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
stride_cnmf = 10  # amount of overlap between the patches in pixels (20)
K = 20  # number of components per patch (10)
gSig = [6, 6]
# initialization method (if analyzing dendritic data using 'sparse_nmf')
method_init = 'greedy_roi'
ssub = 2  # spatial subsampling during initialization
tsub = 2  # temporal subsampling during intialization

# parameters for component evaluation
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

#%% whole pipeline

# set parameters
# dataset dependent parameters
fr = 30  # imaging rate in frames per second
decay_time = 0.4  # length of a typical transient in seconds (0.4)
dxy = (0.83, 0.76)  # spatial resolution in x and y in (um per pixel) [(1.66, 1.52) for 1x, (0.83, 0.76) for 2x]
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
gnb = 3  # number of global background components (3)
merge_thr = 0.70  # merging threshold, max correlation allowed (0.86)
rf = 25
# half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
stride_cnmf = 10  # amount of overlap between the patches in pixels (20)
K = 20  # number of components per patch (10)
gSig = [10, 10]
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
min_SNR = 6  # signal to noise ratio for accepting a component (default 2)
SNR_lowest = 3
rval_thr = 0.7  # space correlation threshold for accepting a component (default 0.85)
cnn_thr = 0.4  # threshold for CNN based classifier (default 0.99)
cnn_lowest = 0.01  # neurons with cnn probability lower than this value are rejected (default 0.1)

opts_dict = {'decay_time': decay_time,
             'SNR_lowest': SNR_lowest,
             'min_SNR': min_SNR,
             'rval_thr': rval_thr,
             'use_cnn': True,
             'min_cnn_thr': cnn_thr,
             'cnn_lowest': cnn_lowest}
opts = opts.change_params(params_dict=opts_dict)

# place cell parameters
params = {'root': None,                  # main directory of this session
          'trans_length': 0.5,           # minimum length in seconds of a significant transient
          'trans_thresh': 4,             # factor of sigma above which a transient is significant
          'bin_length': 2,               # length in cm VR distance of each bin in which to group the dF/F traces (has to be divisor of track_length
          'bin_window_avg': 3,           # sliding window of bins (left and right) for trace smoothing
          'bin_base': 0.25,              # fraction of lowest bins that are averaged for baseline calculation
          'place_thresh': 0.25,          # threshold of being considered for place fields, calculated
                                         # from difference between max and baseline dF/F
          'min_pf_size_cm': 10,          # minimum size in cm for a place field (should be 15-20 cm)
          'fluo_infield': 7,             # factor above which the mean DF/F in the place field should lie compared to outside the field
          'trans_time': 0.2,             # fraction of the (unbinned!) signal while the mouse is located in
                                         # the place field that should consist of significant transients
          'track_length': 400,           # length in cm of the virtual reality corridor
          'split_size': 10}              # size in frames of bootstrapping segments

root = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M22\20191129a\N1'
pipe.whole_caiman_pipeline(root, opts, params, dview)

#%% separate functions
cnm = pipe.run_source_extraction(images, opts, dview=dview)
#if images.shape[0] > 40000:
#    lcm = pipe.get_local_correlation(images[::2])
#else:
lcm = pipe.get_local_correlation(images)
cnm.estimates.Cn = lcm
cnm.estimates.plot_contours(img=lcm)
#plt.savefig(os.path.join(root, 'all_components.png'))
#plt.close()

#%% COMPONENT EVALUATION
# the components are evaluated in three ways:
#   a) the shape of each component must be correlated with the data
#   b) a minimum peak SNR is required over the length of a transient
#   c) each shape passes a CNN based classifier
min_SNR = 6  # signal to noise ratio for accepting a component (default 2)
SNR_lowest = 3
rval_thr = 0.7  # space correlation threshold for accepting a component (default 0.85)
cnn_thr = 0.4  # threshold for CNN based classifier (default 0.99)
cnn_lowest = 0.01  # neurons with cnn probability lower than this value are rejected (default 0.1)

cnm.params.set('quality', {'decay_time': decay_time,
                           'SNR_lowest': SNR_lowest,
                           'min_SNR': min_SNR,
                           'rval_thr': rval_thr,
                           'use_cnn': True,
                           'min_cnn_thr': cnn_thr,
                           'cnn_lowest': cnn_lowest})
cnm = pipe.run_evaluation(images, cnm, dview=dview)

cnm.estimates.plot_contours(img=cnm.estimates.Cn, idx=cnm.estimates.idx_components, display_numbers=False)
plt.savefig(os.path.join(root, 'eval_components.png'))
pipe.save_cnmf(cnm, root=root)
print(f'Finished with {root}!')

cnm.estimates.view_components(images, img=cnm.estimates.Cn,
                               idx=cnm.estimates.idx_components)
cnm.estimates.view_components(images, img=cnm.estimates.Cn,
                               idx=cnm.estimates.idx_components_bad)
cnm.estimates.select_components(use_object=True)
cnm.estimates.detrend_df_f(quantileMin=8, frames_window=500)
cnm.estimates.view_components(img=cnm.estimates.Cn)

#pipe.save_cnmf(cnm, root=root)
#%% Initialize PlaceCellFinder object

#cnm = pipe.load_cnmf(root)
params = {'root': root,                  # main directory of this session
          'trans_length': 0.5,           # minimum length in seconds of a significant transient
          'trans_thresh': 4,             # factor of sigma above which a transient is significant
          'bin_length': 2,               # length in cm VR distance of each bin in which to group the dF/F traces (has to be divisor of track_length
          'bin_window_avg': 3,           # sliding window of bins (left and right) for trace smoothing
          'bin_base': 0.25,              # fraction of lowest bins that are averaged for baseline calculation
          'place_thresh': 0.25,          # threshold of being considered for place fields, calculated
                                         # from difference between max and baseline dF/F
          'min_pf_size_cm': 10,          # minimum size in cm for a place field (should be 15-20 cm)
          'fluo_infield': 7,             # factor above which the mean DF/F in the place field should lie compared to outside the field
          'trans_time': 0.2,             # fraction of the (unbinned!) signal while the mouse is located in
                                         # the place field that should consist of significant transients
          'track_length': 400,           # length in cm of the virtual reality corridor
          'split_size': 10}              # size in frames of bootstrapping segments

pcf = pc.PlaceCellFinder(cnm, params)

# split traces into trials
pcf.split_traces_into_trials()
# create significant-transient-only traces
pcf.create_transient_only_traces()
# align the frames to the VR position using merged behavioral data
pcf.import_behavior_and_align_traces()
# look for place cells
pcf.find_place_cells()
pcf.plot_all_place_cells()
# save pcf object
pcf.save()
#%%

