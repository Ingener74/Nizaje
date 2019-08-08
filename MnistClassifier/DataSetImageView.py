#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from PySide2.QtGui import QPainter
from PySide2.QtWidgets import QWidget


class DataSetImageView(FigureCanvas):
    def __init__(self, parent=None, width=5, height=5, dpi=100):

        self.figure = Figure((width, height), dpi)
        self.axis = self.figure.add_subplot(111)

        self.set_data(np.array([
            [1, 0, 1],
            [0, 1, 1],
            [0, 1, 0]
        ], 'float32'))

        super(DataSetImageView, self).__init__(self.figure)
        self.setParent(parent)

    def set_data(self, data: np.ndarray):

        self.figure.clf()
        self.axis = self.figure.add_subplot(111)
        self.axis.imshow(data)
        self.figure.canvas.draw()

