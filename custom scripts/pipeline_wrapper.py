from caiman.source_extraction import cnmf
import place_cell_pipeline as pipe
import behavior_import as behavior
import place_cell_class as pc

"""
Complete pipeline for place cell analysis, from motion correction to place cell finding

Condition before starting:
Tiff and behavioral files (encoder, position, frames/licks) have to be in separate folders for each trial, and trial
folders have to be grouped in one folder per field of view. Several FOVs can be grouped in a folder per session.
"""

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
    'border_nan': 'copy'
}

opts = cnmf.params.CNMFParams(params_dict=mc_dict)


#%% Set working directory

root = pipe.set_file_paths()

#%% Perform motion correction
roots = [r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M18\20191123a',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191120',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191121b',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191122a',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M19\20191123a',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M22\20191119a',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M22\20191120',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M22\20191121b',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M22\20191122a',
         r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M22\20191123a']

for root in roots:
    if root == r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M22\20191119a':
        opts.change_params({'dxy': (0.83, 0.76)})
    motion_file = pipe.motion_correction(root, opts, remove_f_order=True)

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
behavior.align_behavior(root, performance_check=True, overwrite=True, verbose=True)

# evaluate behavior
mouse_list = []
#%% CaImAn source extraction

#%% Initialize PlaceCellFinder object

cnm = cnmf.cnmf.load_CNMF(r'E:\PhD\Data\DG\M14_20191014\N2\N2_results.hdf5')
params = {'root': root,                  # main directory of this session
          'trial_list': folder_list,     # list of all trial directories in this session
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
          'track_length': 170,           # length in cm of the virtual reality corridor
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
#%%

