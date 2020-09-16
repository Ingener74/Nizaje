#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import numpy as np
from PIL import Image
from PySide2.QtCore import Qt
from PySide2.QtGui import QKeyEvent, QPixmap
from PySide2.QtWidgets import QWidget, QFileDialog
from tensorflow.keras.models import load_model, Model

from CatDogClassifier.Ui_MainWidget import Ui_MainWidget


class MainWidget(QWidget):
    def __init__(self, parent=None):
        super(MainWidget, self).__init__(parent)
        self.ui = Ui_MainWidget()
        self.ui.setupUi(self)

        self.ai: Model

        self.look_for_models()

        self.ui.pushButtonOpen.clicked.connect(self.open_image)
        self.ui.comboBoxModel.currentIndexChanged.connect(self.model_changed)

        self.open_image_from_file_name('res/cat.png')

    def look_for_models(self):
        for root, dirs, files in os.walk('.'):
            for file_name in files:
                if file_name.endswith('.h5'):
                    self.ui.comboBoxModel.addItem(file_name)

        if self.ui.comboBoxModel.count() > 0:
            self.ui.comboBoxModel.setCurrentIndex(0)
            self.ai = self.load_model(self.ui.comboBoxModel.itemText(0))
            self.ai.summary()

    def model_changed(self, index):
        self.ai = self.load_model(self.ui.comboBoxModel.itemText(index))
        self.ai.summary()

    @staticmethod
    def load_model(model_name):
        return load_model(model_name)

    def open_image(self):
        image_file_name, filters = QFileDialog.getOpenFileName(None, 'Open image file')
        if image_file_name == '':
            return
        self.open_image_from_file_name(image_file_name)

    def open_image_from_file_name(self, file_name):
        image = QPixmap()
        image.load(file_name)
        image = image.scaled(300, 300)
        self.ui.labelInputImage.setPixmap(image)
        cat = Image.open(file_name)
        cat = cat.resize((150, 150))
        cat = np.asarray(cat)
        if cat.shape[2] == 4:
            cat = cat[:, :, :-1]
        cat = cat.reshape((1, cat.shape[0], cat.shape[1], cat.shape[2]))
        if self.ai is not None:
            output = self.ai.predict(cat)
            self.ui.horizontalSliderOutput.setValue(output[0] * 100)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Escape:
            self.close()
        else:
            event.ignore()
