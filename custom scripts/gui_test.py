import cv2
import numpy as np
import pyqtgraph as pg
import scipy
from matplotlib import pyplot as plt
from pyqtgraph import FileDialog
from pyqtgraph.Qt import QtGui
from pyqtgraph.parametertree import Parameter, ParameterTree
import os

import caiman as cm
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.source_extraction.cnmf.params import CNMFParams

# Always start by initializing Qt (only once per application)
app = QtGui.QApplication([])

try:
    cv2.setNumThreads(1)
except:
    print('Open CV is naturally single threaded')


## Define a top-level widget to hold everything
w = QtGui.QWidget()

## Create some widgets to be placed inside
btn = QtGui.QPushButton('press me')
text = QtGui.QLineEdit('enter text')
win = pg.GraphicsLayoutWidget()
win.setMaximumWidth(300)
win.setMinimumWidth(200)
hist = pg.HistogramLUTItem() # Contrast/color control
win.addItem(hist)
p1 = pg.PlotWidget()
p2 = pg.PlotWidget()
p3 = pg.PlotWidget()
t = ParameterTree()
t_action = ParameterTree()
action_layout = QtGui.QGridLayout()


## Create a grid layout to manage the widgets size and position
layout = QtGui.QGridLayout()
w.setLayout(layout)

# A plot area (ViewBox + axes) for displaying the image
#p1 = win.addPlot(title="Image here")
# Item for displaying image data
img = pg.ImageItem()
p1.addItem(img)

img2 = pg.ImageItem()
p3.addItem(img2)

hist.setImageItem(img)

# Draggable line for setting isocurve level (for setting contour threshold)
thrshcomp_line = pg.InfiniteLine(angle=0, movable=True, pen='g')
hist.vb.addItem(thrshcomp_line)
hist.vb.setMouseEnabled(y=False) # makes user interaction a little easier
thrshcomp_line.setValue(100)
thrshcomp_line.setZValue(1000) # bring iso line above contrast controls


## Add widgets to the layout in their proper positions
layout.addWidget(win, 1, 0)   # histogram
layout.addWidget(p3, 0, 2)   # denoised movie

layout.addWidget(t, 0, 0)   # upper-right table
layout.addWidget(t_action, 1, 2)  # bottom-right table
layout.addWidget(p1, 0, 1)  # raw movie
layout.addWidget(p2, 1, 1)  # calcium trace window


#enable only horizontal zoom for the traces component
p2.setMouseEnabled(x=True, y=False)


# draw something in the raw-movie field and set the histogram borders correspondingly
test_img_file = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\PhD Meetings\20200106\M18_prestroke.png'
test_img = plt.imread(test_img_file)

img.setImage(np.rot90(test_img[:, :, 0],3))
hist.setLevels(test_img[:, :, 0].min(), test_img[:, :, 0].max())


p2.setMouseEnabled(x=True, y=False)


# Another plot area for displaying ROI data
p2.setMaximumHeight(250)


# set position and scale of image
img.scale(1, 1)

# zoom to fit image
p1.autoRange()


mode = "reset"
p2.setTitle("mode: %s" % (mode))


## Display the widget as a new window
w.show()

## Start the Qt event loop
app.exit(app.exec_())