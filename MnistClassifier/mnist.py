#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

from PySide2.QtWidgets import QApplication

from Utils import setup_gpu_memory_growth
from MnistClassifier.MainWidget import MainWidget
from MnistClassifier.SelectInputFromDataSet import SelectInputFromDataSet


def main():
    setup_gpu_memory_growth()

    app = QApplication()

    widget = MainWidget()
    select_input_from_dataset = SelectInputFromDataSet()

    select_input_from_dataset.set_image_signal.connect(widget.ui.widgetInputDraw.set_image)
    widget.ui.pushButtonSelectFromDataSet.clicked.connect(select_input_from_dataset.show)

    widget.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
