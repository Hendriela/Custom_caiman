#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 09:33:09 2019
@author: adhoff
"""
# Imports
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import TextBox
import numpy as np


### Read input of command line arguments or use default values
default_mouse = 8
default_day = 3

parser = argparse.ArgumentParser(description="Select neurons from correlation image.")

parser.add_argument(
    '-m', '--mouse', help="Integer which uniquely identifies one mouse",
    type=int, default=default_mouse
    )
parser.add_argument(
    '-d', '--day', help="Integer which uniquely identifies one day of recording",
    type=int, default=default_day
    )
arguments = parser.parse_args()


### Connect to DataJoint pipeline and load data
login.connectToDatabase()
from schema import img

key = {'mouse_id': arguments.mouse, 'day': arguments.day}
selected = img.Imaging() & key

# load data from pipeline
avg = (img.MotionCorrection() & selected).fetch1('template')
cor = (img.MotionCorrection() & selected).fetch1('overall_correlation_map')

###  Plot average and correlation image

# open figure
fig = plt.figure()
plt.clf()
fig.canvas.set_window_title('SELECT NEURONS  Mouse: {}  Day: {}'.format(key['mouse_id'], key['day']))

gs = gridspec.GridSpec(2, 2, height_ratios=[10,1])
ax0 = plt.subplot( gs[0,0])
ax1 = plt.subplot( gs[0,1], sharex=ax0, sharey=ax0)
txt = plt.subplot( gs[1,0:2])

# text_box = TextBox(txt, 'Status:', initial='Click to select neurons, backspace to remove last neuron, enter when you are done!')
txt.text(x=0.3,y=0.1,s='Click to select neurons, backspace to remove last neuron, enter when you are done!')
txt.get_xaxis().set_visible(False)
txt.get_yaxis().set_visible(False)
txt.get_xaxis().set_ticks([])
txt.get_yaxis().set_ticks([])

# Average image
minA, maxA = np.percentile(avg, [5, 99.9])
im0 = ax0.imshow( avg , vmin=minA, vmax=maxA)
ax0.set_title('Average image')

# Correlation image
minC, maxC = np.percentile(cor, [5, 99.9])
im1 = ax1.imshow( cor , vmin=minC, vmax=maxC)
ax1.set_title('Correlation image')

# set adjustable colorbar
from plotting import plot_utils

cbar0 = fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
cbar0 = plot_utils.DraggableColorbar(cbar0,ax0, im0, vmin=minA,vmax=maxA)
cbar0.connect()

cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1 = plot_utils.DraggableColorbar(cbar1,ax1, im1, vmin=minC,vmax=maxC)
cbar1.connect()

### Detect click events

# class from https://stackoverflow.com/questions/48446351/distinguish-button-press-event-from-drag-and-zoom-clicks-in-matplotlib
class Click():
    def __init__(self, axes, func, second_axes = None, button=1):
        self.ax=axes
        self.ax1 = second_axes
        self.func=func
        self.button=button
        self.press=False
        self.move = False
        self.c1=self.ax.figure.canvas.mpl_connect('button_press_event', self.onpress)
        self.c2=self.ax.figure.canvas.mpl_connect('button_release_event', self.onrelease)
        self.c3=self.ax.figure.canvas.mpl_connect('motion_notify_event', self.onmove)

    def onclick(self,event):
        # print('Click')
        if event.inaxes == self.ax or event.inaxes == self.ax1 :
            if event.button == self.button:
                # trigger the defined event function
                self.func(event, self.ax, self.ax1)

    def onpress(self,event):
        self.press=True
    def onmove(self,event):
        if self.press:
            self.move=True
    def onrelease(self,event):
        if self.press and not self.move:
            self.onclick(event)
        self.press=False; self.move=False


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
click = Click(axes=ax0, func=func, second_axes=ax1, button=1)


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
        file_name = 'tmp_data/clickedNeurons_mouse_{}_day_{}.txt'.format(arguments.mouse,
                                                                 arguments.day)

        with open(file_name, 'w') as file:
            # write header
            file.write('Neuron\tX\tY\n')
            # write neuron positions in each line
            i = 0
            for x, y in zip(pos_x, pos_y):
                file.write('{}\t{}\t{}\n'.format(i,x,y) )
                i += 1

        print('Wrote positions of {} neurons to file {}'.format(i,file_name))
        txt.clear()
        txt.text(x=0.3,y=0.1,s='Wrote positions of {} neurons to file {}'.format(i,file_name))
        # text_box = TextBox(txt, 'Status:', initial='Wrote positions of {} neurons to file {}'.format(i,file_name))
        ax0.figure.canvas.draw()


# connect keyboard events
fig.canvas.mpl_connect('key_press_event', press)

# show the created figure and wait till figure is closed
# plt.tight_layout()
plt.show()