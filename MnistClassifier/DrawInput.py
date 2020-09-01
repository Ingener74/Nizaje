#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from PySide2.QtCore import Qt, QPoint, Signal
from PySide2.QtGui import QPainter, QBrush, QColor, QMouseEvent
from PySide2.QtWidgets import QWidget


class DrawInput(QWidget):
    input_changed = Signal(np.ndarray)

    def __init__(self, parent=None):
        super(DrawInput, self).__init__(parent)

        self.cell_count = (28, 28)
        self.cell_size = (10, 10)

        self.data = np.zeros((self.cell_count[0], self.cell_count[1]), dtype=float)

        self.setMaximumSize(self.cell_size[1] * self.cell_count[1], self.cell_size[0] * self.cell_count[0])
        self.setMinimumSize(self.cell_size[1] * self.cell_count[1], self.cell_size[0] * self.cell_count[0])

    def set_image(self, image: np.ndarray):
        if image.dtype != np.dtype(np.float):
            self.data = image.astype('float64') / 255.0
        else:
            self.data = image
        self.repaint()
        self.input_changed.emit(self.data)

    def get_image(self):
        return self.data

    def clear(self):
        self.set_image(np.zeros((self.cell_count[0], self.cell_count[1]), dtype=float))

    def paintEvent(self, event):
        painter = QPainter(self)

        for y in range(self.cell_count[0]):
            for x in range(self.cell_count[1]):
                painter.setPen(QColor(127, 127, 127))
                v = self.data[y, x]
                painter.setBrush(QBrush(QColor(v * 255, v * 255, v * 255)))
                h = self.cell_size[1]
                w = self.cell_size[0]
                painter.drawRect(x * h, y * w, h, w)

        painter.end()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.draw(event.pos())

    def mouseMoveEvent(self, event: QMouseEvent):
        self.draw(event.pos())

    def draw(self, pos: QPoint):
        y = int(pos.y() / self.cell_size[0])
        x = int(pos.x() / self.cell_size[1])
        assert (0 <= y < self.cell_count[0])
        assert (0 <= x < self.cell_count[1])
        self.data[y, x] = 1.0
        self.repaint()
        self.input_changed.emit(self.data)
