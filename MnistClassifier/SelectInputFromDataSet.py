#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from PySide2.QtCore import Qt, Signal
from PySide2.QtWidgets import QWidget
from PySide2.QtGui import QKeyEvent, QShowEvent
from tensorflow.keras.datasets import mnist
import numpy as np

from Ui_SelectInputFromDataSet import Ui_SelectInputFromDataSet


class SelectInputFromDataSet(QWidget):
    set_image_signal = Signal(np.ndarray)

    def __init__(self, parent=None):
        super(SelectInputFromDataSet, self).__init__(parent)

        self.ui = Ui_SelectInputFromDataSet()
        self.ui.setupUi(self)

        self.ui.pushButtonSet.clicked.connect(self.set_image)
        self.ui.spinBox.valueChanged.connect(self.spin_changed)

        self.test_images = None

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Escape:
            self.hide()
        else:
            event.ignore()

    def showEvent(self, event: QShowEvent):
        (_, _), (self.test_images, test_labels) = mnist.load_data()
        self.spin_changed(0)

    def spin_changed(self, number):
        self.ui.widgetDataSetImage.set_data(self.test_images[number])

    def set_image(self):
        image = self.test_images[self.ui.spinBox.value()]
        self.set_image_signal.emit(image)
        self.hide()
