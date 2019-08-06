#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from keras.models import Model, load_model
from PySide2.QtCore import Qt
from PySide2.QtGui import QKeyEvent
from PySide2.QtWidgets import QApplication, QWidget

from Ui_MainWidget import Ui_MainWidget


class MainWidget(QWidget):
    def __init__(self, parent=None):
        super(MainWidget, self).__init__(parent)

        self.ui = Ui_MainWidget()
        self.ui.setupUi(self)

        for root, dirs, files in os.walk('.', ):
            for f in files:
                if f.endswith('.h5'):
                    self.ui.comboBoxModel.addItem(f)

        assert(self.ui.comboBoxModel.count() > 0)
        self.neuron_net: Model = load_model(self.ui.comboBoxModel.itemText(0))

        self.ui.widgetInputDraw.input_changed.connect(self.input_changed)
        self.ui.pushButtonPredict.clicked.connect(self.predict)
        self.ui.pushButtonClear.clicked.connect(self.ui.widgetInputDraw.clear)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Escape:
            self.close()
        else:
            event.ignore()

    def predict(self):
        image = self.ui.widgetInputDraw.get_image()
        predict_image = image.reshape(1, image.shape[0] * image.shape[1])
        result = self.neuron_net.predict(predict_image)
        self.ui.widgetHistogram.set_data(result)

    def input_changed(self, image):
        if self.ui.checkBoxInteractive.isChecked():
            predict_image = image.reshape(1, image.shape[0] * image.shape[1])
            result = self.neuron_net.predict(predict_image)
            self.ui.widgetHistogram.set_data(result)
