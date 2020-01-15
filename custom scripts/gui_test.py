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

def test():

    def make_color_img(img, gain=255, min_max=None,out_type=np.uint8):
        if min_max is None:
            min_ = img.min()
            max_ = img.max()
        else:
            min_, max_ = min_max

        img = (img-min_)/(max_-min_)*gain
        img = img.astype(out_type)
        img = np.dstack([img]*3)
        return img

    data=None
    path=r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M18\20191121b\N4'
    ### FIND DATA ###
    if data is None:    # different conditions on file loading (not necessary if data was already provided)
        if path is None:    # if no path has been given, open a window prompt to select a directory
            F = FileDialog()

            # load object saved by CNMF
            path = F.getExistingDirectory(caption='Select folder from which to load a PCF or CNMF file')

        try:    # first try to get CNMF data from a PCF object (should be most up-to-date)
            cnm_obj = pipe.load_pcf(path).cnmf
        except FileNotFoundError:
            try:
                cnm_obj = pipe.load_cnmf(path)
            except FileNotFoundError:
                raise FileNotFoundError(f'Could not find data to load in {path}!')
    else:
        cnm_obj = data


    def draw_contours_overall(md):
        if md is "reset":
            draw_contours()
        elif md is "neurons":
            if neuron_selected is True:
                #if a specific neuron has been selected, only one contour should be changed while thrshcomp_line is changing
                if nr_index is 0:
                    #if user does not start to move through the frames
                    draw_contours_update(estimates.background_image, img)
                    draw_contours_update(comp2_scaled, img2)
                else:
                    # NEVER CALLED IN THIS VERSION WITHOUT MMAP SINCE NR_INDEX NEVER CHANGES (NO NR_VLINE)
                    draw_contours_update(raw_mov_scaled, img)
                    draw_contours_update(frame_denoise_scaled, img2)
            else:
                #if no specific neuron has been selected, all the contours are changing
                draw_contours()
        else:
            #md is "background":
            return


    def draw_contours():
        global thrshcomp_line, estimates, img
        bkgr_contours = estimates.background_image.copy()

        if len(estimates.idx_components) > 0:
            contours = [cv2.findContours(cv2.threshold(img, np.int(thrshcomp_line.value()), 255, 0)[1], cv2.RETR_TREE,
                                         cv2.CHAIN_APPROX_SIMPLE)[0] for img in estimates.img_components[estimates.idx_components]]
            SNRs = np.array(estimates.r_values)
            iidd = np.array(estimates.idx_components)

            idx1 = np.where(SNRs[iidd] < 0.1)[0]
            idx2 = np.where((SNRs[iidd] >= 0.1) &
                            (SNRs[iidd] < 0.25))[0]
            idx3 = np.where((SNRs[iidd] >= 0.25) &
                            (SNRs[iidd] < 0.5))[0]
            idx4 = np.where((SNRs[iidd] >= 0.5) &
                            (SNRs[iidd] < 0.75))[0]
            idx5 = np.where((SNRs[iidd] >= 0.75) &
                            (SNRs[iidd] < 0.9))[0]
            idx6 = np.where(SNRs[iidd] >= 0.9)[0]

            cv2.drawContours(bkgr_contours, sum([contours[jj] for jj in idx1], []), -1, (255, 0, 0), 1)
            cv2.drawContours(bkgr_contours, sum([contours[jj] for jj in idx2], []), -1, (0, 255, 0), 1)
            cv2.drawContours(bkgr_contours, sum([contours[jj] for jj in idx3], []), -1, (0, 0, 255), 1)
            cv2.drawContours(bkgr_contours, sum([contours[jj] for jj in idx4], []), -1, (255, 255, 0), 1)
            cv2.drawContours(bkgr_contours, sum([contours[jj] for jj in idx5], []), -1, (255, 0, 255), 1)
            cv2.drawContours(bkgr_contours, sum([contours[jj] for jj in idx6], []), -1, (0, 255, 255), 1)

        img.setImage(bkgr_contours, autoLevels=False)
    # pg.setConfigOptions(imageAxisOrder='row-major')


    def draw_contours_update(cf, im):
        global thrshcomp_line, estimates
        curFrame = cf.copy()

        if len(estimates.idx_components) > 0:
            contours = [cv2.findContours(cv2.threshold(img, np.int(thrshcomp_line.value()), 255, 0)[1], cv2.RETR_TREE,
                                         cv2.CHAIN_APPROX_SIMPLE)[0] for img in estimates.img_components[estimates.idx_components]]
            SNRs = np.array(estimates.r_values)
            iidd = np.array(estimates.idx_components)

            idx1 = np.where(SNRs[iidd] < 0.1)[0]
            idx2 = np.where((SNRs[iidd] >= 0.1) &
                            (SNRs[iidd] < 0.25))[0]
            idx3 = np.where((SNRs[iidd] >= 0.25) &
                            (SNRs[iidd] < 0.5))[0]
            idx4 = np.where((SNRs[iidd] >= 0.5) &
                            (SNRs[iidd] < 0.75))[0]
            idx5 = np.where((SNRs[iidd] >= 0.75) &
                            (SNRs[iidd] < 0.9))[0]
            idx6 = np.where(SNRs[iidd] >= 0.9)[0]

            if min_dist_comp in idx1:
                cv2.drawContours(curFrame, contours[min_dist_comp], -1, (255, 0, 0), 1)
            if min_dist_comp in idx2:
                cv2.drawContours(curFrame, contours[min_dist_comp], -1, (0, 255, 0), 1)
            if min_dist_comp in idx3:
                cv2.drawContours(curFrame, contours[min_dist_comp], -1, (0, 0, 255), 1)
            if min_dist_comp in idx4:
                cv2.drawContours(curFrame, contours[min_dist_comp], -1, (255, 255, 0), 1)
            if min_dist_comp in idx5:
                cv2.drawContours(curFrame, contours[min_dist_comp], -1, (255, 0, 255), 1)
            if min_dist_comp in idx6:
                cv2.drawContours(curFrame, contours[min_dist_comp], -1, (0, 255, 255), 1)

        im.setImage(curFrame, autoLevels=False)

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
    test_img_file = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\M18\20191121b\N4\local_correlation_image.png'
    test_img = plt.imread(test_img_file)

    img.setImage(np.rot90(test_img[:, :, 0], 3))
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
    app.exec_()

