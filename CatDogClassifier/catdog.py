#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from PySide2.QtWidgets import QApplication

from MainWidget import MainWidget


def main():
    app = QApplication(sys.argv)

    main_widget = MainWidget()
    main_widget.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
