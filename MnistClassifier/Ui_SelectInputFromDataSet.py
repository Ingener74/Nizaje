# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'res/SelectInputFromDataSet.ui',
# licensing of 'res/SelectInputFromDataSet.ui' applies.
#
# Created: Tue Aug  6 15:16:20 2019
#      by: pyside2-uic  running on PySide2 5.13.0
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_SelectInputFromDataSet(object):
    def setupUi(self, SelectInputFromDataSet):
        SelectInputFromDataSet.setObjectName("SelectInputFromDataSet")
        SelectInputFromDataSet.resize(384, 397)
        self.verticalLayout = QtWidgets.QVBoxLayout(SelectInputFromDataSet)
        self.verticalLayout.setObjectName("verticalLayout")
        self.widgetDataSetImage = DataSetImageView(SelectInputFromDataSet)
        self.widgetDataSetImage.setObjectName("widgetDataSetImage")
        self.verticalLayout.addWidget(self.widgetDataSetImage)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.spinBox = QtWidgets.QSpinBox(SelectInputFromDataSet)
        self.spinBox.setObjectName("spinBox")
        self.horizontalLayout.addWidget(self.spinBox)
        self.pushButtonSet = QtWidgets.QPushButton(SelectInputFromDataSet)
        self.pushButtonSet.setObjectName("pushButtonSet")
        self.horizontalLayout.addWidget(self.pushButtonSet)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.verticalLayout.setStretch(0, 1)

        self.retranslateUi(SelectInputFromDataSet)
        QtCore.QMetaObject.connectSlotsByName(SelectInputFromDataSet)

    def retranslateUi(self, SelectInputFromDataSet):
        SelectInputFromDataSet.setWindowTitle(QtWidgets.QApplication.translate("SelectInputFromDataSet", "Select input from dataset", None, -1))
        self.pushButtonSet.setText(QtWidgets.QApplication.translate("SelectInputFromDataSet", "Set image", None, -1))

from DataSetImageView import DataSetImageView
