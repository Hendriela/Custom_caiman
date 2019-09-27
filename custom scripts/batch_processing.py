import warnings
import cv2
import glob
import logging
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import os
import sys
import caiman as cm
from caiman.motion_correction import MotionCorrect

from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
sys.path.append(r'C:\Users\hheise\PycharmProjects\Caiman\custom scripts')
import post_analysis as post
import imageio

try:
    cv2.setNumThreads(0)
except:
    pass

try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass
#%% load parameters from a previous analysis
root = r'E:\PhD\Data\CA1\Maus 3 13.08.2019'
file_name = r'file_00011'

opts = load_params(root, file_name)

manual_files = True

if manual_files:
    fnames = [r'E:\PhD\Data\CA1\Maus 3 13.08.2019\file_00011.tif',
              r'E:\PhD\Data\CA1\Maus 3 13.08.2019\file_00012.tif',
              r'E:\PhD\Data\CA1\Maus 3 13.08.2019\file_00013.tif',
              r'E:\PhD\Data\CA1\Maus 3 13.08.2019\file_00014.tif',
              r'E:\PhD\Data\CA1\Maus 3 13.08.2019\file_00015.tif',
              r'E:\PhD\Data\CA1\Maus 3 13.08.2019\file_00016.tif',
              r'E:\PhD\Data\CA1\Maus 3 13.08.2019\file_00017.tif',
              r'E:\PhD\Data\CA1\Maus 3 13.08.2019\file_00018.tif',
              r'E:\PhD\Data\CA1\Maus 3 13.08.2019\file_00019.tif',
              r'E:\PhD\Data\CA1\Maus 3 13.08.2019\file_00020.tif',
              r'E:\PhD\Data\CA1\Maus 3 13.08.2019\file_00021.tif',
              r'E:\PhD\Data\CA1\Maus 3 13.08.2019\file_00022.tif']
else:
    fnames = glob.glob(root+r'\*.tif')

# set up cluster for parallel processing
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)
opts.change_params({'n_processes': n_processes, 'fnames': fnames})

#%% manually set parameters

root = r'E:\PhD\Data\CA1\Maus 3 13.08.2019'
manual_files = True

if manual_files:
    movie_list = [r'E:\PhD\Data\CA1\Maus 3 13.08.2019\file_00011.tif',
              r'E:\PhD\Data\CA1\Maus 3 13.08.2019\file_00012.tif',
              r'E:\PhD\Data\CA1\Maus 3 13.08.2019\file_00013.tif',
              r'E:\PhD\Data\CA1\Maus 3 13.08.2019\file_00014.tif',
              r'E:\PhD\Data\CA1\Maus 3 13.08.2019\file_00015.tif',
              r'E:\PhD\Data\CA1\Maus 3 13.08.2019\file_00016.tif',
              r'E:\PhD\Data\CA1\Maus 3 13.08.2019\file_00017.tif',
              r'E:\PhD\Data\CA1\Maus 3 13.08.2019\file_00018.tif',
              r'E:\PhD\Data\CA1\Maus 3 13.08.2019\file_00019.tif',
              r'E:\PhD\Data\CA1\Maus 3 13.08.2019\file_00020.tif',
              r'E:\PhD\Data\CA1\Maus 3 13.08.2019\file_00021.tif',
              r'E:\PhD\Data\CA1\Maus 3 13.08.2019\file_00022.tif']
else:
    movie_list = glob.glob(root+r'\*.tif')

fr = 30                         # imaging rate in frames per second
decay_time = 3                # length of a typical transient in seconds
dxy = (1, 1)                    # spatial resolution in x and y in (um per pixel)
max_shift_um = (12., 12.)       # maximum shift in um
patch_motion_um = (25., 25.)    # patch size for non-rigid correction in um

# motion correction parameters
pw_rigid = True                 # flag to select rigid vs pw_rigid motion correction
# maximum allowed rigid shift in pixels
max_shifts = [int(a / b) for a, b in zip(max_shift_um, dxy)]
# start a new patch for pw-rigid motion correction every x pixels
strides_mc = tuple([int(a / b) for a, b in zip(patch_motion_um, dxy)])
overlaps = (12, 12)             # overlap between patches (size of patch in pixels: strides+overlaps)
max_deviation_rigid = 3         # maximum deviation allowed for patch with respect to rigid shifts

# CNMF parameters
p = 1                       # order of the autoregressive system
gnb = 3                     # number of global background components
merge_thr = 0.86            # merging threshold, max correlation allowed
rf = 50                     # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
stride_cnmf = 20            # amount of overlap between the patches in pixels
K = 10                       # number of components per patch
gSig = [13, 11]             # expected half size of neurons in pixels
method_init = 'greedy_roi'  # initialization method
ssub = 2                    # spatial subsampling during initialization
tsub = 2                    # temporal subsampling during intialization

# Evaluation parameters
# signal to noise ratio for accepting a component (default 0.5 and 2)
SNR_lowest = 3
min_SNR = 6
# space correlation threshold for accepting a component (default -1 and 0.85)
rval_lowest = -1
rval_thr = 0.75
# threshold for CNN based classifier (default 0.1 and 0.99)
cnn_lowest = 0.1
cnn_thr = 0.99
use_cnn = True

# set up cluster for parallel processing
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

# enter parameters in the dictionary
opts_dict = {'fnames': movie_list[0], 'fr': fr,  'decay_time': decay_time, 'dxy': dxy, 'pw_rigid': pw_rigid,
             'max_shifts': max_shifts, 'strides': strides_mc, 'overlaps': overlaps,
             'max_deviation_rigid': max_deviation_rigid, 'border_nan': 'copy', 'nb': gnb, 'rf': rf, 'K': K, 'gSig': gSig,
             'stride': stride_cnmf, 'method_init': method_init, 'rolling_sum': True, 'merge_thr': merge_thr,
             'n_processes': n_processes,  'only_init': True, 'ssub': ssub, 'tsub': tsub, 'min_SNR': min_SNR,
             'rval_thr': rval_thr, 'min_cnn_thr': cnn_thr, 'cnn_lowest': cnn_lowest, 'use_cnn': True}



#%% Motion correction

# perform batch motion correction
count = 1
mc_files = []
for file in movie_list:
    opts = params.CNMFParams(params_dict=opts_dict)
    file_name = file.split('\\')[-1].split('.')[0]
    mc = MotionCorrect([file], dview=dview, **opts.get_group('motion'))
    print(f'Starting motion correction of {file_name}!')
    mc.motion_correct(save_movie=False)
    border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0
    fname_new = cm.save_memmap(mc.mmap_file, base_name=f'{file_name}_', order='C',
                               border_to_0=border_to_0)  # exclude borders
    mc_files.append(fname_new)
    print(f'Finished motion correction of {file_name}! ({count}/{len(fnames)} complete)\n')
    count += 1

#%% Component detection and evaluation

use_custom_files = True
save_eval_graph = True

if use_custom_files:
    file_list = [r'E:\PhD\Data\CA1\Maus 3 13.08.2019\file_00012_d1_512_d2_512_d3_1_order_C_frames_1801_.mmap']
else:
    file_list = glob.glob(root+r'\*.mmap')
    print(f'Processing all .mmap files in directory {root}')

count = 1
for file in file_list:
    file_name = file.split('\\')[-1].split('.')[0][:10]
    print(f'Starting to process file {file_name}...')

    opts_dict['fnames'] = movie_list[1]
    opts = params.CNMFParams(params_dict=opts_dict)

    Yr, dims, T = cm.load_memmap(file)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')

    cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)

    print(f'\tStarting to fit file {file_name}...')
    opts.change_params({'p': 0})
    cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
    cnm = cnm.fit(images)

    Cn = cm.local_correlations(images, swap_dim=False)
    Cn[np.isnan(Cn)] = 0
    cnm.estimates.plot_contours(img=Cn)
    plt.title('Contour plots of found components')
    plt.savefig(root+f'\{file_name}_detection.png')
    plt.close()

    print('\tDone! \n')

    print(f'\tStarting to re-fit file {file_name}...')
    cnm.params.change_params({'p': p})
    cnm2 = cnm.refit(images, dview=dview)
    print('\tDone! \n')

    cnm2.estimates.evaluate_components(images, params=cnm2.params, dview=dview)
    cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=250)

    if save_eval_graph:
        cnm2.estimates.plot_contours(img=Cn, idx=cnm2.estimates.idx_components)
        plt.savefig(root+f'\{file_name}_evaluation.png')
        plt.close()

    dirname = root + r'\\' + file_name + "_results.hdf5"

    print(f'\tSaving file: {file_name + "_results.hdf5"}...')
    cnm2.estimates.Cn = Cn

    cnm2.save(dirname)
    print(f'Done with file {file_name}! ({count}/{len(file_list)})\n')
    count += 1

    del cnm
    #del cnm2
    del opts

#%% Component alignment and re-registration

file_list = glob.glob(root+r'\*.hdf5')

cnm_list = []
for file in file_list:
    cnm_list.append(cnmf.load_CNMF(file))


#%% helper function that loads previously performed parameters for batch analysis

def load_params(root, file_name):
    dir = root+r'\\'+file_name+r'_results.hdf5'
    cnm = cnmf.load_CNMF(dir)

    return cnm.params
