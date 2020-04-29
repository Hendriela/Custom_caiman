#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 09:33:09 2019
@author: adhoff
"""
# Imports
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import matplotlib.gridspec as gridspec
from datetime import datetime
import preprocess as pre



# ### Read input of command line arguments or use default values
# default_mouse = 8
# default_day = 3
#
# parser = argparse.ArgumentParser(description="Select neurons from correlation image.")
#
# parser.add_argument(
#     '-m', '--mouse', help="Integer which uniquely identifies one mouse",
#     type=int, default=default_mouse
#     )
# parser.add_argument(
#     '-d', '--day', help="Integer which uniquely identifies one day of recording",
#     type=int, default=default_day
#     )
# arguments = parser.parse_args()

root = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M37\20200323'

cor = io.imread(root + r'\local_correlation_image.tif')
avg = io.imread(root + r'\mean_intensity_image.tif')

###  Plot average and correlation image
fig = plt.figure()
plt.clf()
curr_sess = datetime.strptime(os.path.split(root)[1], '%Y%m%d').strftime('%d.%m.%Y')
curr_mouse = os.path.split(os.path.split(root)[0])[1]
fig.canvas.set_window_title(f'SELECT NEURONS  Mouse: {curr_mouse}  Day: {curr_sess}')

gs = gridspec.GridSpec(2, 2, height_ratios=[10, 1])
ax0 = plt.subplot(gs[0, 0])
ax1 = plt.subplot(gs[0, 1], sharex=ax0, sharey=ax0)
txt = plt.subplot(gs[1, 0:2])

# text_box = TextBox(txt, 'Status:', initial='Click to select neurons, backspace to remove last neuron, enter when you are done!')
txt.text(x=0.3, y=0.1, s='Click to select neurons, backspace to remove last neuron, enter when you are done!')
txt.get_xaxis().set_visible(False)
txt.get_yaxis().set_visible(False)
txt.get_xaxis().set_ticks([])
txt.get_yaxis().set_ticks([])

# Average image
minA, maxA = np.percentile(avg, [5, 99.9])
im0 = ax0.imshow(avg, vmin=minA, vmax=maxA)
ax0.set_title('Average image')

# Correlation image
minC, maxC = np.percentile(cor, [5, 99.9])
im1 = ax1.imshow(cor, vmin=minC, vmax=maxC)
ax1.set_title('Correlation image')

# set adjustable colorbar

cbar0 = fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
cbar0 = pre.DraggableColorbar(cbar0, ax0, im0, vmin=minA, vmax=maxA)
cbar0.connect()

cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1 = pre.DraggableColorbar(cbar1, ax1, im1, vmin=minC, vmax=maxC)
cbar1.connect()


def drawCross(ax, x=20, y=20):
    ax.axvline(x=x, color='w', LineWidth = 0.5)
    ax.axhline(y=y, color='w', LineWidth = 0.5)

crosses = list()
crosses1 = list()
pos_x = list()
pos_y = list()

def func(event, ax, ax1):
    x = int(event.xdata)
    y = int(event.ydata)

    # print('Click location:\tX: ',x , '\tY: ', y)

    # draw a cross at the click location of axis (also ax1 if not None)
    cross = ax.plot(x, y, 'x', color='red')
    cross1 = ax1.plot(x, y, 'x', color='red')

    crosses.append(cross)
    crosses1.append(cross1)
    pos_x.append(x)
    pos_y.append(y)

    ax.figure.canvas.draw()


# connect click events to avg and cor axes
click = pre.Click(axes=ax0, func=func, second_axes=ax1, button=1)


### Read keyboard enter and backspace events
def press(event):
    if event.key == 'backspace':
        print('Removing last element')

        # remove last element from list
        pos_x.pop()
        pos_y.pop()
        # remove also from plot and refresh plot
        crosses.pop()[0].remove()
        crosses1.pop()[0].remove()
        ax0.figure.canvas.draw()

    if event.key == 'enter':
        ### Save the clicked positions to file
        file_name = os.path.join(root, 'clicked_neurons_{}.txt'.format(datetime.now().strftime('%Y%m%d_%H%M%S')))

        with open(file_name, 'w') as file:
            # write header
            file.write('Neuron\tX\tY\n')
            # write neuron positions in each line
            i = 0
            for x, y in zip(pos_x, pos_y):
                file.write('{}\t{}\t{}\n'.format(i, x, y))
                i += 1

        print('Wrote positions of {} neurons to file {}'.format(i, file_name))
        txt.clear()
        txt.text(x=0.3, y=0.1, s='Wrote positions of {} neurons to file {}'.format(i, file_name))
        # text_box = TextBox(txt, 'Status:', initial='Wrote positions of {} neurons to file {}'.format(i,file_name))
        ax0.figure.canvas.draw()


# connect keyboard events
fig.canvas.mpl_connect('key_press_event', press)

# show the created figure and wait till figure is closed
# plt.tight_layout()
plt.show()
