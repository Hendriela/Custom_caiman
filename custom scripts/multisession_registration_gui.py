import numpy as np
import pyqtgraph as pg
import scipy
from pyqtgraph import FileDialog
from pyqtgraph.Qt import QtGui
from PyQt5 import QtWidgets

def gui():

    app = QtWidgets.QApplication([])
    w = QtWidgets.QWidget()
    ref_win = pg.PlotWidget()
    comp_win = QtWidgets.QWidget()

    ref_cell = pg.ImageItem()
    ref_win.addItem(ref_cell)

    comp_cell_1 = QtWidgets.QPushButton('1')
    comp_cell_2 = QtWidgets.QPushButton('2')
    comp_cell_3 = QtWidgets.QPushButton('3')
    comp_cell_4 = QtWidgets.QPushButton('4')
    comp_cell_5 = QtWidgets.QPushButton('5')
    comp_cell_6 = QtWidgets.QPushButton('6')
    comp_cell_7 = QtWidgets.QPushButton('7')
    comp_cell_8 = QtWidgets.QPushButton('8')
    comp_cell_9 = QtWidgets.QPushButton('9')



    h_layout = QtWidgets.QHBoxLayout(w)

    h_layout.addWidget(ref_win)
    h_layout.addWidget(comp_win)



    w.show()
    app.exit(app.exec_())
