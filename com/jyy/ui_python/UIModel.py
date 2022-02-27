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
        self.back.setGeometry(QtCore.QRect(90, 690, 101, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.back.setFont(font)
        self.back.setObjectName("back")
        self.next = QtWidgets.QPushButton(self.centralwidget)
        self.next.setGeometry(QtCore.QRect(380, 690, 101, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.next.setFont(font)
        self.next.setObjectName("next")
        self.graphicsViewLeft = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsViewLeft.setGeometry(QtCore.QRect(0, 110, 571, 581))
        self.graphicsViewLeft.setObjectName("graphicsViewLeft")
        self.graphicsViewRight = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsViewRight.setGeometry(QtCore.QRect(670, 110, 561, 581))
        self.graphicsViewRight.setObjectName("graphicsViewRight")
        self.lightSlider = QtWidgets.QSlider(self.centralwidget)
        self.lightSlider.setGeometry(QtCore.QRect(400, 740, 431, 21))
        self.lightSlider.setMaximum(300)
        self.lightSlider.setOrientation(QtCore.Qt.Horizontal)
        self.lightSlider.setObjectName("lightSlider")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(340, 730, 51, 41))
        self.label.setObjectName("label")
        self.rotateThate = QtWidgets.QLabel(self.centralwidget)
        self.rotateThate.setGeometry(QtCore.QRect(340, 760, 51, 41))
        self.rotateThate.setObjectName("rotateThate")
        self.rotateSlider = QtWidgets.QSlider(self.centralwidget)
        self.rotateSlider.setGeometry(QtCore.QRect(400, 770, 431, 21))
        self.rotateSlider.setMinimum(-360)
        self.rotateSlider.setMaximum(360)
        self.rotateSlider.setSingleStep(10)
        self.rotateSlider.setOrientation(QtCore.Qt.Horizontal)
        self.rotateSlider.setObjectName("rotateSlider")
        self.leftRightButton = QtWidgets.QPushButton(self.centralwidget)
        self.leftRightButton.setGeometry(QtCore.QRect(580, 342, 81, 31))
        self.leftRightButton.setObjectName("leftRightButton")
        self.rightLeftButton = QtWidgets.QPushButton(self.centralwidget)
        self.rightLeftButton.setGeometry(QtCore.QRect(580, 420, 81, 31))
        self.rightLeftButton.setObjectName("rightLeftButton")
        self.showCount = QtWidgets.QLabel(self.centralwidget)
        self.showCount.setGeometry(QtCore.QRect(260, 692, 61, 41))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.showCount.setFont(font)
        self.showCount.setAlignment(QtCore.Qt.AlignCenter)
        self.showCount.setObjectName("showCount")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(180, 70, 71, 31))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(940, 60, 91, 41))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setEnabled(True)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1233, 22))
        self.menubar.setDefaultUp(True)
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setEnabled(True)
        self.menu.setGeometry(QtCore.QRect(206, 114, 119, 126))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.menu.setFont(font)
        self.menu.setObjectName("menu")
        self.menu_10 = QtWidgets.QMenu(self.menu)
        self.menu_10.setObjectName("menu_10")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setGeometry(QtCore.QRect(246, 114, 114, 158))
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
        self.menu_9 = QtWidgets.QMenu(self.menubar)
        self.menu_9.setObjectName("menu_9")
        self.about = QtWidgets.QMenu(self.menubar)
        self.about.setObjectName("about")
        self.menu_8 = QtWidgets.QMenu(self.menubar)
        self.menu_8.setObjectName("menu_8")
        self.other = QtWidgets.QMenu(self.menubar)
        self.other.setObjectName("other")
        MainWindow.setMenuBar(self.menubar)
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
        self.addTextMark = QtWidgets.QAction(MainWindow)
        self.addTextMark.setObjectName("addTextMark")
        self.addGraphMask = QtWidgets.QAction(MainWindow)
        self.addGraphMask.setObjectName("addGraphMask")
        self.history1 = QtWidgets.QAction(MainWindow)
        self.history1.setObjectName("history1")
        self.history2 = QtWidgets.QAction(MainWindow)
        self.history2.setObjectName("history2")
        self.history3 = QtWidgets.QAction(MainWindow)
        self.history3.setObjectName("history3")
        self.closeProcess = QtWidgets.QAction(MainWindow)
        self.closeProcess.setObjectName("closeProcess")
        self.openProcess = QtWidgets.QAction(MainWindow)
        self.openProcess.setObjectName("openProcess")
        self.thresholdValue = QtWidgets.QAction(MainWindow)
        self.thresholdValue.setObjectName("thresholdValue")
        self.FeatureExtraction = QtWidgets.QAction(MainWindow)
        self.FeatureExtraction.setObjectName("FeatureExtraction")
        self.ImgClassfiyAndRcong = QtWidgets.QAction(MainWindow)
        self.ImgClassfiyAndRcong.setObjectName("ImgClassfiyAndRcong")
        self.dilate = QtWidgets.QAction(MainWindow)
        self.dilate.setObjectName("dilate")
        self.erode = QtWidgets.QAction(MainWindow)
        self.erode.setObjectName("erode")
        self.xingtaiixuetidu = QtWidgets.QAction(MainWindow)
        self.xingtaiixuetidu.setObjectName("xingtaiixuetidu")
        self.dingmao = QtWidgets.QAction(MainWindow)
        self.dingmao.setObjectName("dingmao")
        self.heimao = QtWidgets.QAction(MainWindow)
        self.heimao.setObjectName("heimao")
        self.menu_10.addAction(self.history1)
        self.menu_10.addAction(self.history2)
        self.menu_10.addAction(self.history3)
        self.menu.addAction(self.menu_10.menuAction())
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
        self.menu_9.addAction(self.closeProcess)
        self.menu_9.addAction(self.openProcess)
        self.menu_9.addAction(self.dilate)
        self.menu_9.addAction(self.erode)
        self.menu_9.addAction(self.xingtaiixuetidu)
        self.menu_9.addAction(self.dingmao)
        self.menu_9.addAction(self.heimao)
        self.menu_8.addAction(self.addTextMark)
        self.other.addAction(self.thresholdValue)
        self.other.addAction(self.FeatureExtraction)
        self.other.addAction(self.ImgClassfiyAndRcong)
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())
        self.menubar.addAction(self.menu_3.menuAction())
        self.menubar.addAction(self.menu_4.menuAction())
        self.menubar.addAction(self.menu_5.menuAction())
        self.menubar.addAction(self.menu_6.menuAction())
        self.menubar.addAction(self.menu_7.menuAction())
        self.menubar.addAction(self.menu_9.menuAction())
        self.menubar.addAction(self.menu_8.menuAction())
        self.menubar.addAction(self.other.menuAction())
        self.menubar.addAction(self.about.menuAction())

        self.retranslateUi(MainWindow)
        # self.fileopen.triggered.connect(self.fileopen.trigger)
        # self.exit.triggered.connect(self.exit.trigger)
        # self.filesave.triggered.connect(self.filesave.trigger)
        MainWindow.destroyed.connect(MainWindow.update)
        # self.classfic.triggered.connect(self.classfic.trigger)
        # self.recognize.triggered.connect(self.recognize.trigger)
        # self.poissonNoise.triggered.connect(self.poissonNoise.trigger)
        # self.splotNoise.triggered.connect(self.splotNoise.trigger)
        # self.jiaoYanNoise.triggered.connect(self.jiaoYanNoise.trigger)
        # self.GaussNoise.triggered.connect(self.GaussNoise.trigger)
        # self.RadomTransformer.triggered.connect(self.RadomTransformer.trigger)
        # self.DCTTransformer.triggered.connect(self.DCTTransformer.trigger)
        # self.fourierTransform.triggered.connect(self.fourierTransform.trigger)
        # self.printScreen.triggered.connect(self.printScreen.trigger)
        # self.rotate.triggered.connect(self.rotate.trigger)
        # self.lightLevel.triggered.connect(self.lightLevel.trigger)
        # self.grayLevel.triggered.connect(self.grayLevel.trigger)
        # self.zoomDown.triggered.connect(self.zoomDown.trigger)
        # self.zoomUp.triggered.connect(self.zoomUp.trigger)
        # self.HSVColorModel.triggered.connect(self.HSVColorModel.trigger)
        # self.YCbCrColorModel.triggered.connect(self.YCbCrColorModel.trigger)
        # self.NTSCColorModel.triggered.connect(self.NTSCColorModel.trigger)
        # self.histogramBalance.triggered.connect(self.histogramBalance.trigger)
        # self.colorReinforce.triggered.connect(self.colorReinforce.trigger)
        # self.noColorRenforce.triggered.connect(self.noColorRenforce.trigger)
        # self.BHistgram.triggered.connect(self.BHistgram.trigger)
        # self.GHistogram.triggered.connect(self.GHistogram.trigger)
        # self.RHistogram.triggered.connect(self.RHistogram.trigger)
        # self.sharpenFilterNoLinear.triggered.connect(self.sharpenFilterNoLinear.trigger)
        # self.sharpenFilterLinear.triggered.connect(self.sharpenFilterLinear.trigger)
        # self.smoothFilterNoLinear.triggered.connect(self.smoothFilterNoLinear.trigger)
        # self.smoothFilterLinear.triggered.connect(self.smoothFilterLinear.trigger)
        # self.lowPassFilter.triggered.connect(self.lowPassFilter.trigger)
        # self.highPassFilter.triggered.connect(self.highPassFilter.trigger)
        # self.next.clicked.connect(self.next.click)
        # self.back.clicked.connect(self.back.click)
        # self.history1.triggered.connect(self.history1.trigger)
        # self.history2.triggered.connect(self.history2.trigger)
        # self.history3.triggered.connect(self.history3.trigger)
        # self.lightSlider.valueChanged['int'].connect(self.lightSlider.update)
        # self.rotateSlider.valueChanged['int'].connect(self.rotateSlider.update)
        # self.closeProcess.triggered.connect(self.closeProcess.trigger)
        # self.openProcess.triggered.connect(self.openProcess.trigger)
        # self.FeatureExtraction.triggered.connect(self.FeatureExtraction.trigger)
        # self.ImgClassfiyAndRcong.triggered.connect(self.ImgClassfiyAndRcong.trigger)
        # self.thresholdValue.triggered.connect(self.thresholdValue.trigger)
        # self.leftRightButton.clicked.connect(self.leftRightButton.click)
        # self.rightLeftButton.clicked.connect(self.rightLeftButton.click)
        # self.dilate.triggered.connect(self.dilate.trigger)
        # self.dingmao.triggered.connect(self.dingmao.trigger)
        # self.erode.triggered.connect(self.erode.trigger)
        # self.xingtaiixuetidu.triggered.connect(self.xingtaiixuetidu.trigger)
        # self.heimao.triggered.connect(self.heimao.trigger)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.back.setText(_translate("MainWindow", "上一张"))
        self.next.setText(_translate("MainWindow", "下一张"))
        self.label.setText(_translate("MainWindow", "  亮度"))
        self.rotateThate.setText(_translate("MainWindow", "  旋转"))
        self.leftRightButton.setText(_translate("MainWindow", "→(移出)"))
        self.rightLeftButton.setText(_translate("MainWindow", "←(移入)"))
        self.showCount.setText(_translate("MainWindow", "1/9"))
        self.label_2.setText(_translate("MainWindow", "操作区"))
        self.label_3.setText(_translate("MainWindow", "显示区"))
        self.menu.setTitle(_translate("MainWindow", "文件"))
        self.menu_10.setTitle(_translate("MainWindow", "最近浏览"))
        self.menu_2.setTitle(_translate("MainWindow", "编辑"))
        self.menu_3.setTitle(_translate("MainWindow", "变换"))
        self.menu_4.setTitle(_translate("MainWindow", "噪声"))
        self.menu_5.setTitle(_translate("MainWindow", "滤波"))
        self.menu_6.setTitle(_translate("MainWindow", "直方图统计"))
        self.menu_7.setTitle(_translate("MainWindow", "图像增强"))
        self.menu_9.setTitle(_translate("MainWindow", "形态学处理"))
        self.about.setTitle(_translate("MainWindow", "关于"))
        self.menu_8.setTitle(_translate("MainWindow", "添加水印"))
        self.other.setTitle(_translate("MainWindow", "其他"))
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
        self.addTextMark.setText(_translate("MainWindow", "添加文字水印"))
        self.addGraphMask.setText(_translate("MainWindow", "添加图片水印"))
        self.history1.setText(_translate("MainWindow", "history1"))
        self.history2.setText(_translate("MainWindow", "history2"))
        self.history3.setText(_translate("MainWindow", "history3"))
        self.closeProcess.setText(_translate("MainWindow", "闭运算"))
        self.openProcess.setText(_translate("MainWindow", "开运算"))
        self.thresholdValue.setText(_translate("MainWindow", "阈值分割"))
        self.FeatureExtraction.setText(_translate("MainWindow", "特征提取"))
        self.ImgClassfiyAndRcong.setText(_translate("MainWindow", "图像分类与识别"))
        self.dilate.setText(_translate("MainWindow", "膨胀"))
        self.erode.setText(_translate("MainWindow", "腐蚀"))
        self.xingtaiixuetidu.setText(_translate("MainWindow", "形态学梯度"))
        self.dingmao.setText(_translate("MainWindow", "顶帽"))
        self.heimao.setText(_translate("MainWindow", "黑帽"))