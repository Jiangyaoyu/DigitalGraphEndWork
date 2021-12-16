from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
import cv2 as cv
import numpy as np
import sys
import os
import traceback
import random

from com.jyy.ui.UIModel import Ui_MainWindow


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        # 禁用窗口大小变换
        self.setWindowFlags(QtCore.Qt.WindowCloseButtonHint)
        self.setFixedSize(self.width(), self.height())

        # 设置组件属性
        self.setWindowTitle('数字图像实验演示平台')

        # 方法绑定
        self.fileopen.triggered.connect(self.filesave.trigger)
        self.recentBrowse.triggered.connect(self.recentBrowse.trigger)
        self.exit.triggered.connect(self.exit.trigger)
        self.filesave.triggered.connect(self.filesave.trigger)
        self.classfic.triggered.connect(self.classfic.trigger)
        self.recognize.triggered.connect(self.recognize.trigger)
        self.poissonNoise.triggered.connect(self.poissonNoise.trigger)
        self.splotNoise.triggered.connect(self.splotNoise.trigger)
        self.jiaoYanNoise.triggered.connect(self.jiaoYanNoise.trigger)
        self.GaussNoise.triggered.connect(self.GaussNoise.trigger)
        self.RadomTransformer.triggered.connect(self.RadomTransformer.trigger)
        self.DCTTransformer.triggered.connect(self.DCTTransformer.trigger)
        self.fourierTransform.triggered.connect(self.fourierTransform.trigger)
        self.printScreen.triggered.connect(self.printScreen.trigger)
        self.rotate.triggered.connect(self.rotate.trigger)
        self.lightLevel.triggered.connect(self.lightLevel.trigger)
        self.grayLevel.triggered.connect(self.grayLevel.trigger)
        self.zoomDown.triggered.connect(self.zoomDown.trigger)
        self.zoomUp.triggered.connect(self.zoomUp.trigger)
        self.HSVColorModel.triggered.connect(self.HSVColorModel.trigger)
        self.YCbCrColorModel.triggered.connect(self.YCbCrColorModel.trigger)
        self.NTSCColorModel.triggered.connect(self.NTSCColorModel.trigger)
        self.histogramBalance.triggered.connect(self.histogramBalance.trigger)
        self.colorReinforce.triggered.connect(self.colorReinforce.trigger)
        self.noColorRenforce.triggered.connect(self.noColorRenforce.trigger)
        self.BHistgram.triggered.connect(self.BHistgram.trigger)
        self.GHistogram.triggered.connect(self.GHistogram.trigger)
        self.RHistogram.triggered.connect(self.RHistogram.trigger)
        self.sharpenFilterNoLinear.triggered.connect(self.sharpenFilterNoLinear.trigger)
        self.sharpenFilterLinear.triggered.connect(self.sharpenFilterLinear.trigger)
        self.smoothFilterNoLinear.triggered.connect(self.smoothFilterNoLinear.trigger)
        self.smoothFilterLinear.triggered.connect(self.smoothFilterLinear.trigger)
        self.lowPassFilter.triggered.connect(self.lowPassFilter.trigger)
        self.highPassFilter.triggered.connect(self.highPassFilter.trigger)
        self.next.clicked.connect(self.next.click)
        self.back.clicked.connect(self.back.click)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    mainWindon = MainWindow()
    mainWindon.show()

    sys.exit(app.exec_())
