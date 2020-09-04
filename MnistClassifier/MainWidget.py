#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import importlib
import sys
from io import StringIO
from tensorflow.keras.models import Model, load_model
from PySide2.QtCore import Qt
from PySide2.QtGui import QKeyEvent
from PySide2.QtWidgets import QWidget, QMessageBox

from Ui_MainWidget import Ui_MainWidget


class MainWidget(QWidget):
    def __init__(self, parent=None):
        super(MainWidget, self).__init__(parent)

        self.ui = Ui_MainWidget()
        self.ui.setupUi(self)

        self.convert_func = None

        for root, dirs, files in os.walk('.', ):
            for f in files:
                if f.endswith('.h5'):
                    self.ui.comboBoxModel.addItem(f)

        assert (self.ui.comboBoxModel.count() > 0)

        self.neuron_net: Model = load_model(self.ui.comboBoxModel.itemText(0))
        self.import_converter(self.ui.comboBoxModel.itemText(0))

        self.ui.widgetInputDraw.input_changed.connect(self.input_changed)
        self.ui.pushButtonPredict.clicked.connect(self.on_click_predict)
        self.ui.pushButtonClear.clicked.connect(self.ui.widgetInputDraw.clear)
        self.ui.pushButtonModelInfo.clicked.connect(self.show_model_info)

        self.ui.comboBoxModel.currentIndexChanged.connect(self.change_model)

    def import_converter(self, model_file_name):
        convert_name = model_file_name[:-3]

        convert_module = importlib.import_module(convert_name)

        self.convert_func = convert_module.convert

    def show_model_info(self):
        summary_string = StringIO()
        self.neuron_net.summary(print_fn=lambda x: summary_string.write(x + '\n'))
        summary = summary_string.getvalue()
        QMessageBox.information(self, 'Model information', summary)

    def change_model(self, index):
        self.neuron_net = load_model(self.ui.comboBoxModel.itemText(index))
        self.import_converter(self.ui.comboBoxModel.itemText(index))

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Escape:
            self.close()
        else:
            event.ignore()

    def on_click_predict(self):
        image = self.ui.widgetInputDraw.get_image()
        self.predict(image)

    def input_changed(self, image):
        if self.ui.checkBoxInteractive.isChecked():
            self.predict(image)

    def predict(self, image):
        assert (self.convert_func is not None)

        input_ = self.convert_func(image)
        try:
            output_ = self.neuron_net.predict(input_)
        except Exception as e:
            print(e)
        except:
            print("Unexpected error:", sys.exc_info()[0])
        else:
            self.ui.widgetHistogram.set_data(output_)
