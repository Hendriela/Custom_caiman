#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from caiman.source_extraction.cnmf import cnmf
import caiman as cm
import tkinter as tk
from tkinter import filedialog


#%%

# TODO: INSERT CNM2.ESTIMATES EVERYWHERE TO PASS DOWN DATA

def save_results(run_name=''): # basically useless, cnmf.save does the same
    ### SAVES CNMF, DECONV AND CNN RESULTS ####
    # If called before estimates.select_components, all components with corresponding idx_comp are saved. #
    # If called afterwards, all available components (only good ones) are saved. #

    # comp_available = True  # assume that component selection is available
    dirname = fnames[0][:-4] + "_analysis"
    os.makedirs(os.path.dirname(dirname), exist_ok=True)  # create directory for caiman results if necessary

    timestamp = str(datetime.now())
    curr_time = timestamp[:4] + timestamp[5:7] + timestamp[8:10] + '_' + timestamp[11:13] + timestamp[
                                                                                            14:16] + timestamp[17:19]
    if run_name:
        run_dir = dirname + '/' + run_name
    else:
        run_dir = dirname + '/run_' + curr_time  # set runname to timestamp if none was provided
    os.makedirs(os.path.dirname(run_dir), exist_ok=True)  # create directory for current run

    # save parameters of the current run
    filename = run_dir + '/params.txt'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as param_file:
        param_file.write(str(cnm2.params))

    # all components are being saved, together with indices of good and bad components to untangle it later
    # save denoised and deconvolved neural activity
    filename = run_dir + '/denoise_deconv_act.gz'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savetxt(filename, cnm2.estimates.C)
    # save df/f normalized neural activity
    filename = run_dir + '/df_f_norm.gz'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savetxt(filename, cnm2.estimates.F_dff)
    # save deconvolved spikes
    filename = run_dir + '/spikes.gz'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savetxt(filename, cnm2.estimates.S)
    if cnm2.estimates.idx_components:
        # save good/bad component indices
        filename = run_dir + '/idx_comp_good.txt'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.savetxt(filename, cnm2.estimates.idx_components)
        print('Good and bad components with idx_comp_good indices are saved!\n')
    else:
        print('Only good components are saved!')


def load_results():
    file_path = get_filename()
    return cnmf.load_CNMF(file_path)


def load_mcorr_mmap():
    file_path = get_filename()
    Yr, dims, T = cm.load_memmap(file_path)
    return np.reshape(Yr.T, [T] + list(dims), order='F')


def save_movie(estimates, imgs, path, include_bck=True, frame_range=slice(None, None, None), bpx=0, gain=2, include_orig=False, include_res=False):
    dims = imgs.shape[1:]
    if path[:-4] != '.tif':
        raise Exception('File has to be saved as a .tif!')
        return
    if 'movie' not in str(type(imgs)):
        imgs = cm.movie(imgs)
    Y_rec = estimates.A.dot(estimates.C[:, frame_range])
    Y_rec = Y_rec.reshape(dims + (-1,), order='F')
    Y_rec = Y_rec.transpose([2, 0, 1])
    if estimates.W is not None:
        ssub_B = int(round(np.sqrt(np.prod(dims) / estimates.W.shape[0])))
        B = imgs[frame_range].reshape((-1, np.prod(dims)), order='F').T - \
            estimates.A.dot(estimates.C[:, frame_range])
        if ssub_B == 1:
            B = estimates.b0[:, None] + estimates.W.dot(B - estimates.b0[:, None])
        else:
            B = estimates.b0[:, None] + (np.repeat(np.repeat(estimates.W.dot(
                downscale(B.reshape(dims + (B.shape[-1],), order='F'),
                          (ssub_B, ssub_B, 1)).reshape((-1, B.shape[-1]), order='F') -
                downscale(estimates.b0.reshape(dims, order='F'),
                          (ssub_B, ssub_B)).reshape((-1, 1), order='F'))
                                                             .reshape(((dims[0] - 1) // ssub_B + 1, (dims[1] - 1) // ssub_B + 1, -1), order='F'),
                                                             ssub_B, 0), ssub_B, 1)[:dims[0], :dims[1]].reshape((-1, B.shape[-1]), order='F'))
        B = B.reshape(dims + (-1,), order='F').transpose([2, 0, 1])
    elif estimates.b is not None and estimates.f is not None:
        B = estimates.b.dot(estimates.f[:, frame_range])
        if 'matrix' in str(type(B)):
            B = B.toarray()
        B = B.reshape(dims + (-1,), order='F').transpose([2, 0, 1])
    else:
        B = np.zeros_like(Y_rec)
    if bpx > 0:
        B = B[:, bpx:-bpx, bpx:-bpx]
        Y_rec = Y_rec[:, bpx:-bpx, bpx:-bpx]
        imgs = imgs[:, bpx:-bpx, bpx:-bpx]

    if include_orig:
        if include_res:
            # save all three (original, reconstructed and residual data)
            Y_res = imgs[frame_range] - Y_rec - B
            mov = cm.concatenate((imgs[frame_range] - (not include_bck) * B, Y_rec + include_bck * B, Y_res * gain),
                                 axis=2)
        else:
            # save original and reconstructed
            mov = cm.concatenate((imgs[frame_range] - (not include_bck) * B, Y_rec + include_bck * B), axis=2)
    elif include_res:
        # save reconstructed and residual
        Y_res = imgs[frame_range] - Y_rec - B
        mov = cm.concatenate((Y_rec + include_bck * B, Y_res * gain), axis=2)
    else:
        # save only reconstructed
        mov = Y_rec * gain + include_bck * B

    print('Created movie file, now saving...\n')
    if 'movie' not in str(type(mov)):
        mov = cm.movie(mov)
    mov.save(path)
    print(f'Movie successfully saved to {path}!')


def save_local_correlations(img, path):
    if path[:-4] != '.tif':
        raise Exception('File has to be saved as a .tif!')
        return
    if img.shape[1] == img.shape[2] and img.shape[1] != img.shape[0]:
        swap_dim = False
    elif img.shape[0] == img.shape[1] and img.shape[0] != img.shape[2]:
        swap_dim = True
    else:
        raise Exception('Movie dimensions could not be resolved.')
        return

    Cn = cm.local_correlations(img,swap_dim=swap_dim)
    # todo: get the image in a good file format (png is working but bad quality, tif is weird
    Cn.save()


#%% cross correlation test

def cross_test():
    comp_24 = cnm2.estimates.F_dff[23]
    comp_24_shift = np.concatenate((np.zeros(10),comp_24[:-10]))

    comp_24_sub = comp_24.copy()
    comp_24_sub[550:650] = 0
    comp_24_sub[260:320] = 0
    comp_24_sub[90:160] = 0
    comp_24_sub = np.concatenate((np.zeros(100),comp_24_sub[:-100]))

    comp_24_sub[450:550] = comp_24[50:150]

    test_fig, test_ax = plt.subplots(3)
    test_ax[0].plot(comp_24)
    test_ax[1].plot(comp_24_sub)

    # calculate the cross-correlation
    npts = len(comp_24)
    sample_rate = 1 / 30  # in Hz
    # lags = np.arange(start=-(npts*sample_rate)+sample_rate, stop=npts*sample_rate-sample_rate, step=sample_rate)
    lags = np.arange(-npts + 1, npts)
    # remove sample means
    comp_24_dm = comp_24 - comp_24.mean()
    comp_24_shift_dm = comp_24_shift - comp_24_shift.mean()
    comp_24_sub_dm = comp_24_sub - comp_24_sub.mean()
    # calculate correlation
    #trace_cov = np.correlate(comp_24_dm, comp_24_shift_dm, 'full')
    trace_cov = np.correlate(comp_24_dm, comp_24_sub_dm, 'full')
    # normalize against std
    #trace_corr = trace_cov / (npts * comp_24.std() * comp_24_shift.std())
    trace_corr = trace_cov / (npts * comp_24.std() * comp_24_sub.std())

    test_ax[2].plot(trace_corr)
    test_ax[2].set_ylim((-0.1,1))

    np.where(trace_corr == np.max(trace_corr))


def get_filename():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename()
    root.update()
    return path

