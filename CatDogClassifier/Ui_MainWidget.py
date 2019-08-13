# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'res/MainWidget.ui',
# licensing of 'res/MainWidget.ui' applies.
#
# Created: Fri Aug  9 17:06:36 2019
#      by: pyside2-uic  running on PySide2 5.13.0
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_MainWidget(object):
    def setupUi(self, MainWidget):
        MainWidget.setObjectName("MainWidget")
        MainWidget.resize(373, 568)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(MainWidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.comboBoxModel = QtWidgets.QComboBox(MainWidget)
        self.comboBoxModel.setObjectName("comboBoxModel")
        self.verticalLayout_2.addWidget(self.comboBoxModel)
        self.groupBox = QtWidgets.QGroupBox(MainWidget)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout.setObjectName("verticalLayout")
        self.labelInputImage = QtWidgets.QLabel(self.groupBox)
        self.labelInputImage.setText("")
        self.labelInputImage.setPixmap(QtGui.QPixmap(":/cat.png"))
        self.labelInputImage.setObjectName("labelInputImage")
        self.verticalLayout.addWidget(self.labelInputImage)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButtonOpen = QtWidgets.QPushButton(self.groupBox)
        self.pushButtonOpen.setObjectName("pushButtonOpen")
        self.horizontalLayout_2.addWidget(self.pushButtonOpen)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.verticalLayout_2.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(MainWidget)
        self.groupBox_2.setObjectName("groupBox_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.groupBox_2)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.groupBox_2)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.horizontalSliderOutput = QtWidgets.QSlider(self.groupBox_2)
        self.horizontalSliderOutput.setEnabled(False)
        self.horizontalSliderOutput.setSliderPosition(50)
        self.horizontalSliderOutput.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSliderOutput.setObjectName("horizontalSliderOutput")
        self.horizontalLayout.addWidget(self.horizontalSliderOutput)
        self.label_2 = QtWidgets.QLabel(self.groupBox_2)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.verticalLayout_2.addWidget(self.groupBox_2)

        self.retranslateUi(MainWidget)
        QtCore.QMetaObject.connectSlotsByName(MainWidget)

    def retranslateUi(self, MainWidget):
        MainWidget.setWindowTitle(QtWidgets.QApplication.translate("MainWidget", "Cat vs. Dog", None, -1))
        self.groupBox.setTitle(QtWidgets.QApplication.translate("MainWidget", "Input", None, -1))
        self.pushButtonOpen.setText(QtWidgets.QApplication.translate("MainWidget", "Открыть", None, -1))
        self.groupBox_2.setTitle(QtWidgets.QApplication.translate("MainWidget", "Output", None, -1))
        self.label.setText(QtWidgets.QApplication.translate("MainWidget", "Cat", None, -1))
        self.label_2.setText(QtWidgets.QApplication.translate("MainWidget", "Dog", None, -1))

import resources_rc
