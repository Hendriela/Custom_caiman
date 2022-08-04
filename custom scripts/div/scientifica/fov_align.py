#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 15/07/2022 12:58
@author: hheise

Function for semi-automatic FOV alignment from a previously acquired stack of the same FOV.
"""

### IMPORTS
import matplotlib
# if matplotlib.get_backend() != 'TkAgg': # If TkAgg is set before during import, Python crashes with a stack overflow
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import tifffile as tiff
import numpy as np
from skimage.registration import phase_cross_correlation
import tkinter as tk
from tkinter import filedialog, messagebox

### LOAD IMAGES
# Open TkInter dialog window on top
root = tk.Tk()
root.wm_attributes('-topmost', 1)
root.withdraw()
# Ask the user for the reference stack and current FOV file paths, and read files into memory
stack_name = filedialog.askopenfilename(parent=root, title='Choose the reference stack', filetypes=(('TIFF', '*.tif'),))
fov_name = filedialog.askopenfilename(parent=root, title='Choose the current FOV', filetypes=(('TIFF', '*.tif'),))
stack = tiff.imread(stack_name)
fov = tiff.imread(fov_name)

# Get stack info from file name
try:
    # frames_per_stack = int(stack_name.split('_')[-3])       # Number of frames per slice, which will be averaged
    # n_slices = stack.shape[0]//frames_per_stack             # Number of slices in the stack
    z_dist = int(stack_name.split('_')[-2][:-2])              # Distance between slices (thickness) in um
    target_slice = int(stack_name.split('_')[-4]) - 1         # Index of the slice which is the imaging plane
    zoom = int(stack_name.split('_')[-1].split('.')[0][0])    # Zoom of the stack and FOV
except:
    messagebox.showerror('Failed to interpret stack info', 'Failed to interpret stack info.\n'
                                                           'The stack file name has to follow this pattern:\n'
                                                           '*_{imaging plane slice idx}_{slice thickness}um_{frames/slice}_{zoom}.tif')
    raise ValueError('Failed to interpret stack info')

# Convert resolution
if zoom == 1:
    res = (1.66, 1.52)
elif zoom == 2:
    res = (0.83, 0.76)
elif zoom == 3:
    res = (0.56, 0.54)
else:
    messagebox.showerror('Error', f'Could not interpret zoom value {zoom}.')
    raise ValueError(f'Could not interpret zoom value {zoom}.')

# Average stacks
# stack_avg = [np.mean(stack[i*frames_per_stack:i*frames_per_stack+frames_per_stack], axis=0) for i in range(n_slices)]
if len(fov.shape) == 3:
    fov_avg = np.mean(fov, axis=0)
elif len(fov.shape) == 2:
    fov_avg = fov
else:
    raise IndexError('Provide 2D or 3D FOV image.')

# Compute phase cross correlation and pixel offset for each frame in the stack
shifts = []
errors = []
diffphases = []
for frame in stack:
    shift, error, _ = phase_cross_correlation(frame, fov_avg)
    shifts.append(shift * res)      # Scale pixel-wise shift with microscope resolution to get shift in um
    errors.append(error)

# Compute results
best_slice = np.argmin(errors)      # The best fitting slice is the one with the lowest error
z_shift = (best_slice - target_slice) * z_dist  # The z-shift is the distance between the target and the best slice
x_shift = shifts[best_slice][0]     # Separate X and Y shifts for plotting
y_shift = shifts[best_slice][1]
z = np.arange(start=-(target_slice*z_dist), stop=(len(stack)-target_slice)*z_dist, step=z_dist)

# Plot results
fig = plt.figure()
x_line, y_line = plt.plot(z, shifts)   # Plot x and y shifts against stack depth
plt.xlabel('depth from target plane[um]')
plt.ylabel('X/Y shifts')
plt.twinx()                         # Plot error on a secondary Y-axis
plt.axvline(z[target_slice], color='red')
plt.ylabel('Correlation error')
error_line = plt.plot(z, errors, color='green')
plt.legend([x_line, y_line, error_line[0]], ['x (left-right)', 'y (top-bottom)', 'error'], loc='lower right')

# Print out shift results in a textbox in the current axes
ax = fig.gca()
results = 'Recommended stage corrections:\nx={:6.2f} um\ny={:6.2f} um\nz={:7d} um'.format(x_shift, y_shift, z_shift)
if z_shift > 20:
    results += '\nWarning: Large z-shift!\nX and Y shifts may be unreliable.\nCorrect Z and take new image.'
plt.text(0.95, 0.25, results, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)
plt.tight_layout()
plt.show()
