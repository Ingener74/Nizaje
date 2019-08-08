#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from PySide2.QtWidgets import QApplication

from SelectInputFromDataSet import SelectInputFromDataSet
from MainWidget import MainWidget

def main():
    app = QApplication(sys.argv)

    widget = MainWidget()
    select_input_from_dataset = SelectInputFromDataSet()
    
    select_input_from_dataset.set_image_signal.connect(widget.ui.widgetInputDraw.set_image)
    widget.ui.pushButtonSelectFromDataSet.clicked.connect(select_input_from_dataset.show)

    widget.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
