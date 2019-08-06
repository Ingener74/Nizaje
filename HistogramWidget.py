#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from PySide2.QtGui import QPainter
from PySide2.QtWidgets import QWidget


class HistogramWidget(FigureCanvas):
    def __init__(self, parent=None, width=5, height=5, dpi=100):

        self.figure = Figure((width, height), dpi)
        self.axis = self.figure.add_subplot(111)

        self.data = np.array([[0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0]], 'float32')

        self.set_data(np.array([[
            0, 0, 0, 0, 0, 
            0, 0, 0, 0, 0
        ]], 'float32'))

        super(HistogramWidget, self).__init__(self.figure)
        self.setParent(parent)

        self.ani = FuncAnimation(self.figure, self.animate, )

    def animate(self, i):
        self.check_data(self.data)

        self.axis.clear()
        indexes = list(range(0, 10))
        self.axis.bar(range(0, 10), self.data[0])
        self.axis.set_xticks(indexes)
        self.axis.set_xticklabels(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))

    def set_data(self, data: np.ndarray):
        self.check_data(data)
        self.data = data

    @staticmethod
    def check_data(data: np.ndarray):
        assert(len(data.shape) == 2)
        assert(data.shape[0] == 1)
        assert(data.shape[1] == 10)
        assert(data.dtype == np.dtype('float32'))
        
        # self.figure.clf()
        # self.axis = self.figure.add_subplot(111)
        # indexes = list(range(0, 10))
        # self.axis.bar(range(0, 10), data[0])
        # self.axis.set_xticks(indexes)
        # self.axis.set_xticklabels(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))
        # self.figure.canvas.draw()
