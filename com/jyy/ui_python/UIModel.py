# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UIModel.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(1233, 897)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.graphicsViewCenter = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsViewCenter.setGeometry(QtCore.QRect(240, 150, 721, 521))
        self.graphicsViewCenter.setObjectName("graphicsViewCenter")
        self.back = QtWidgets.QPushButton(self.centralwidget)
        self.back.setGeometry(QtCore.QRect(400, 690, 101, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.back.setFont(font)
        self.back.setObjectName("back")
        self.next = QtWidgets.QPushButton(self.centralwidget)
        self.next.setGeometry(QtCore.QRect(730, 690, 101, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.next.setFont(font)
        self.next.setObjectName("next")
        self.graphicsViewLeft = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsViewLeft.setGeometry(QtCore.QRect(0, 110, 501, 581))
        self.graphicsViewLeft.setObjectName("graphicsViewLeft")
        self.graphicsViewRight = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsViewRight.setGeometry(QtCore.QRect(730, 110, 501, 581))
        self.graphicsViewRight.setObjectName("graphicsViewRight")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setEnabled(True)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1233, 23))
        self.menubar.setDefaultUp(True)
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setEnabled(True)
        self.menu.setGeometry(QtCore.QRect(269, 129, 125, 144))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.menu.setFont(font)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setGeometry(QtCore.QRect(309, 129, 120, 182))
        self.menu_2.setObjectName("menu_2")
        self.menu_3 = QtWidgets.QMenu(self.menubar)
        self.menu_3.setObjectName("menu_3")
        self.menu_4 = QtWidgets.QMenu(self.menubar)
        self.menu_4.setObjectName("menu_4")
        self.menu_5 = QtWidgets.QMenu(self.menubar)
        self.menu_5.setObjectName("menu_5")
        self.menu_6 = QtWidgets.QMenu(self.menubar)
        self.menu_6.setObjectName("menu_6")
        self.menu_7 = QtWidgets.QMenu(self.menubar)
        self.menu_7.setObjectName("menu_7")
        self.thresholdValue = QtWidgets.QMenu(self.menubar)
        self.thresholdValue.setObjectName("thresholdValue")
        self.menu_9 = QtWidgets.QMenu(self.menubar)
        self.menu_9.setObjectName("menu_9")
        self.FeatuerExtraction = QtWidgets.QMenu(self.menubar)
        self.FeatuerExtraction.setObjectName("FeatuerExtraction")
        self.menu_11 = QtWidgets.QMenu(self.menubar)
        self.menu_11.setObjectName("menu_11")
        self.about = QtWidgets.QMenu(self.menubar)
        self.about.setObjectName("about")
        self.menu_8 = QtWidgets.QMenu(self.menubar)
        self.menu_8.setObjectName("menu_8")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.fileopen = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.fileopen.setFont(font)
        self.fileopen.setObjectName("fileopen")
        self.filesave = QtWidgets.QAction(MainWindow)
        self.filesave.setObjectName("filesave")
        self.exit = QtWidgets.QAction(MainWindow)
        self.exit.setObjectName("exit")
        self.zoomUp = QtWidgets.QAction(MainWindow)
        self.zoomUp.setObjectName("zoomUp")
        self.zoomDown = QtWidgets.QAction(MainWindow)
        self.zoomDown.setObjectName("zoomDown")
        self.grayLevel = QtWidgets.QAction(MainWindow)
        self.grayLevel.setObjectName("grayLevel")
        self.lightLevel = QtWidgets.QAction(MainWindow)
        self.lightLevel.setObjectName("lightLevel")
        self.rotate = QtWidgets.QAction(MainWindow)
        self.rotate.setObjectName("rotate")
        self.printScreen = QtWidgets.QAction(MainWindow)
        self.printScreen.setObjectName("printScreen")
        self.fourierTransform = QtWidgets.QAction(MainWindow)
        self.fourierTransform.setObjectName("fourierTransform")
        self.DCTTransformer = QtWidgets.QAction(MainWindow)
        self.DCTTransformer.setObjectName("DCTTransformer")
        self.RadomTransformer = QtWidgets.QAction(MainWindow)
        self.RadomTransformer.setObjectName("RadomTransformer")
        self.GaussNoise = QtWidgets.QAction(MainWindow)
        self.GaussNoise.setObjectName("GaussNoise")
        self.jiaoYanNoise = QtWidgets.QAction(MainWindow)
        self.jiaoYanNoise.setObjectName("jiaoYanNoise")
        self.splotNoise = QtWidgets.QAction(MainWindow)
        self.splotNoise.setObjectName("splotNoise")
        self.poissonNoise = QtWidgets.QAction(MainWindow)
        self.poissonNoise.setObjectName("poissonNoise")
        self.highPassFilter = QtWidgets.QAction(MainWindow)
        self.highPassFilter.setObjectName("highPassFilter")
        self.lowPassFilter = QtWidgets.QAction(MainWindow)
        self.lowPassFilter.setObjectName("lowPassFilter")
        self.smoothFilterLinear = QtWidgets.QAction(MainWindow)
        self.smoothFilterLinear.setObjectName("smoothFilterLinear")
        self.smoothFilterNoLinear = QtWidgets.QAction(MainWindow)
        self.smoothFilterNoLinear.setObjectName("smoothFilterNoLinear")
        self.sharpenFilterLinear = QtWidgets.QAction(MainWindow)
        self.sharpenFilterLinear.setObjectName("sharpenFilterLinear")
        self.sharpenFilterNoLinear = QtWidgets.QAction(MainWindow)
        self.sharpenFilterNoLinear.setObjectName("sharpenFilterNoLinear")
        self.RHistogram = QtWidgets.QAction(MainWindow)
        self.RHistogram.setObjectName("RHistogram")
        self.GHistogram = QtWidgets.QAction(MainWindow)
        self.GHistogram.setObjectName("GHistogram")
        self.BHistgram = QtWidgets.QAction(MainWindow)
        self.BHistgram.setObjectName("BHistgram")
        self.noColorRenforce = QtWidgets.QAction(MainWindow)
        self.noColorRenforce.setObjectName("noColorRenforce")
        self.colorReinforce = QtWidgets.QAction(MainWindow)
        self.colorReinforce.setObjectName("colorReinforce")
        self.histogramBalance = QtWidgets.QAction(MainWindow)
        self.histogramBalance.setObjectName("histogramBalance")
        self.NTSCColorModel = QtWidgets.QAction(MainWindow)
        self.NTSCColorModel.setObjectName("NTSCColorModel")
        self.YCbCrColorModel = QtWidgets.QAction(MainWindow)
        self.YCbCrColorModel.setObjectName("YCbCrColorModel")
        self.HSVColorModel = QtWidgets.QAction(MainWindow)
        self.HSVColorModel.setObjectName("HSVColorModel")
        self.classfic = QtWidgets.QAction(MainWindow)
        self.classfic.setObjectName("classfic")
        self.recognize = QtWidgets.QAction(MainWindow)
        self.recognize.setObjectName("recognize")
        self.recentBrowse = QtWidgets.QAction(MainWindow)
        self.recentBrowse.setObjectName("recentBrowse")
        self.addTextMark = QtWidgets.QAction(MainWindow)
        self.addTextMark.setObjectName("addTextMark")
        self.addGraphMask = QtWidgets.QAction(MainWindow)
        self.addGraphMask.setObjectName("addGraphMask")
        self.menu.addAction(self.recentBrowse)
        self.menu.addAction(self.fileopen)
        self.menu.addAction(self.filesave)
        self.menu.addAction(self.exit)
        self.menu_2.addAction(self.zoomUp)
        self.menu_2.addAction(self.zoomDown)
        self.menu_2.addAction(self.grayLevel)
        self.menu_2.addAction(self.lightLevel)
        self.menu_2.addAction(self.rotate)
        self.menu_2.addAction(self.printScreen)
        self.menu_3.addAction(self.fourierTransform)
        self.menu_3.addAction(self.DCTTransformer)
        self.menu_3.addAction(self.RadomTransformer)
        self.menu_4.addAction(self.GaussNoise)
        self.menu_4.addAction(self.jiaoYanNoise)
        self.menu_4.addAction(self.splotNoise)
        self.menu_4.addAction(self.poissonNoise)
        self.menu_5.addAction(self.highPassFilter)
        self.menu_5.addAction(self.lowPassFilter)
        self.menu_5.addAction(self.smoothFilterLinear)
        self.menu_5.addAction(self.smoothFilterNoLinear)
        self.menu_5.addAction(self.sharpenFilterLinear)
        self.menu_5.addAction(self.sharpenFilterNoLinear)
        self.menu_6.addAction(self.RHistogram)
        self.menu_6.addAction(self.GHistogram)
        self.menu_6.addAction(self.BHistgram)
        self.menu_7.addAction(self.noColorRenforce)
        self.menu_7.addAction(self.colorReinforce)
        self.menu_7.addAction(self.histogramBalance)
        self.menu_7.addAction(self.NTSCColorModel)
        self.menu_7.addAction(self.YCbCrColorModel)
        self.menu_7.addAction(self.HSVColorModel)
        self.menu_11.addAction(self.classfic)
        self.menu_11.addAction(self.recognize)
        self.menu_8.addAction(self.addTextMark)
        self.menu_8.addAction(self.addGraphMask)
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())
        self.menubar.addAction(self.menu_3.menuAction())
        self.menubar.addAction(self.menu_4.menuAction())
        self.menubar.addAction(self.menu_5.menuAction())
        self.menubar.addAction(self.menu_6.menuAction())
        self.menubar.addAction(self.menu_7.menuAction())
        self.menubar.addAction(self.thresholdValue.menuAction())
        self.menubar.addAction(self.menu_9.menuAction())
        self.menubar.addAction(self.FeatuerExtraction.menuAction())
        self.menubar.addAction(self.menu_11.menuAction())
        self.menubar.addAction(self.menu_8.menuAction())
        self.menubar.addAction(self.about.menuAction())

        self.retranslateUi(MainWindow)
        MainWindow.destroyed.connect(MainWindow.update)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.back.setText(_translate("MainWindow", "上一张"))
        self.next.setText(_translate("MainWindow", "下一张"))
        self.menu.setTitle(_translate("MainWindow", "文件"))
        self.menu_2.setTitle(_translate("MainWindow", "编辑"))
        self.menu_3.setTitle(_translate("MainWindow", "变换"))
        self.menu_4.setTitle(_translate("MainWindow", "噪声"))
        self.menu_5.setTitle(_translate("MainWindow", "滤波"))
        self.menu_6.setTitle(_translate("MainWindow", "直方图统计"))
        self.menu_7.setTitle(_translate("MainWindow", "图像增强"))
        self.thresholdValue.setTitle(_translate("MainWindow", "阈值分割"))
        self.menu_9.setTitle(_translate("MainWindow", "形态学处理"))
        self.FeatuerExtraction.setTitle(_translate("MainWindow", "特征提取"))
        self.menu_11.setTitle(_translate("MainWindow", "图像分类与识别"))
        self.about.setTitle(_translate("MainWindow", "关于"))
        self.menu_8.setTitle(_translate("MainWindow", "添加水印"))
        self.fileopen.setText(_translate("MainWindow", "打开"))
        self.filesave.setText(_translate("MainWindow", "保存"))
        self.exit.setText(_translate("MainWindow", "退出"))
        self.zoomUp.setText(_translate("MainWindow", "放大"))
        self.zoomDown.setText(_translate("MainWindow", "缩小"))
        self.grayLevel.setText(_translate("MainWindow", "灰度"))
        self.lightLevel.setText(_translate("MainWindow", "亮度"))
        self.rotate.setText(_translate("MainWindow", "旋转"))
        self.printScreen.setText(_translate("MainWindow", "截图"))
        self.fourierTransform.setText(_translate("MainWindow", "傅里叶变换"))
        self.DCTTransformer.setText(_translate("MainWindow", "离散余弦变换"))
        self.RadomTransformer.setText(_translate("MainWindow", "Radom变换"))
        self.GaussNoise.setText(_translate("MainWindow", "高斯噪声"))
        self.jiaoYanNoise.setText(_translate("MainWindow", "椒盐噪声"))
        self.splotNoise.setText(_translate("MainWindow", "斑点噪声"))
        self.poissonNoise.setText(_translate("MainWindow", "泊松噪声"))
        self.highPassFilter.setText(_translate("MainWindow", "高通滤波"))
        self.lowPassFilter.setText(_translate("MainWindow", "低通噪声"))
        self.smoothFilterLinear.setText(_translate("MainWindow", "平滑滤波（线性）"))
        self.smoothFilterNoLinear.setText(_translate("MainWindow", "平滑滤波（非线性)"))
        self.sharpenFilterLinear.setText(_translate("MainWindow", "锐化滤波（线性）"))
        self.sharpenFilterNoLinear.setText(_translate("MainWindow", "锐化滤波（非线性）"))
        self.RHistogram.setText(_translate("MainWindow", "R直方图"))
        self.GHistogram.setText(_translate("MainWindow", "G直方图"))
        self.BHistgram.setText(_translate("MainWindow", "B直方图"))
        self.noColorRenforce.setText(_translate("MainWindow", "伪彩色图像"))
        self.colorReinforce.setText(_translate("MainWindow", "真彩色图像"))
        self.histogramBalance.setText(_translate("MainWindow", "直方图均衡"))
        self.NTSCColorModel.setText(_translate("MainWindow", "NTSC颜色模型"))
        self.YCbCrColorModel.setText(_translate("MainWindow", "YCbCr颜色模型"))
        self.HSVColorModel.setText(_translate("MainWindow", "HSV颜色模型"))
        self.classfic.setText(_translate("MainWindow", "分类"))
        self.recognize.setText(_translate("MainWindow", "识别"))
        self.recentBrowse.setText(_translate("MainWindow", "最近浏览"))
        self.addTextMark.setText(_translate("MainWindow", "添加文字水印"))
        self.addGraphMask.setText(_translate("MainWindow", "添加图片水印"))