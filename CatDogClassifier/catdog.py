#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from PySide2.QtWidgets import QApplication

from CatDogClassifier.MainWidget import MainWidget
from Utils import setup_gpu_memory_growth


def main():
    setup_gpu_memory_growth()

    app = QApplication()

    main_widget = MainWidget()
    main_widget.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
