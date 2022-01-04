from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import *
import sys
import os
from com.jyy.ui_python.UIModel import Ui_MainWindow
import cv2 as cv
import traceback

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        # 禁用窗口大小变换
        self.setWindowFlags(QtCore.Qt.WindowCloseButtonHint)
        self.setFixedSize(self.width(), self.height())

        # 设置组件属性
        self.setWindowTitle('数字图像实验演示平台')

        # 定义并初始化成员变量
        self.current_img = None #当前图片
        self.current_idx = -1 #图片列表索引
        self.target_img = None #
        self.source_img_list = [] #图片列表
        self.cacheTxt = []  # 存放从缓存中读取的图片


        #初始化隐藏组件
        self.next.setHidden(True)
        self.back.setHidden(True)
        self.graphicsViewLeft.setHidden(True)
        self.graphicsViewRight.setHidden(True)
        self.graphicsViewCenter.setHidden(True)
        self.graphicsViewCenter.setStyleSheet("border: 0px;background-color:#F0F0F0")#设置中间显示框的背景色，和边框
        self.imgCache_read()#读取缓存路径
        print("初始化时历史记录个数：",len(self.cacheTxt))
        if len(self.cacheTxt) >0:
            self.history1.setText(self.cacheTxt[0])
        else:
            self.history1.setText("")
        if len(self.cacheTxt) >= 2:
            self.history2.setText(self.cacheTxt[1])
        else:
            self.history2.setVisible(False)
        if len(self.cacheTxt) >=3:
            self.history3.setText(self.cacheTxt[2])
        else:
            self.history3.setVisible(False)

        # 方法绑定
        self.fileopen.triggered.connect(self.open_event)
        self.exit.triggered.connect(self.close_window)
        self.filesave.triggered.connect(self.save_event)
        self.classfic.triggered.connect(self.classfic.trigger)
        self.recognize.triggered.connect(self.recognize.trigger)
        self.poissonNoise.triggered.connect(self.poissonNoise_event)
        self.splotNoise.triggered.connect(self.splotNoise_event)
        self.jiaoYanNoise.triggered.connect(self.jiaoYanNoise_event)
        self.GaussNoise.triggered.connect(self.GaussNoise_event)
        self.RadomTransformer.triggered.connect(self.RadomTransformer_event)
        self.DCTTransformer.triggered.connect(self.DCTTransformer_event)
        self.fourierTransform.triggered.connect(self.fourierTransform_event)
        self.printScreen.triggered.connect(self.printScreen_event)
        self.rotate.triggered.connect(self.rotate_event)
        self.lightLevel.triggered.connect(self.lightLevel_event)
        self.grayLevel.triggered.connect(self.grayLevel_event)
        self.zoomDown.triggered.connect(self.zoomDown_event)
        self.zoomUp.triggered.connect(self.zoomUp_event)
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
        self.history1.triggered.connect(self.setHistory1_event)
        self.history2.triggered.connect(self.setHistory2_event)
        self.history3.triggered.connect(self.setHistory3_event)

    #start 噪声

    #高斯噪声
    def GaussNoise_event(self):
        print("高斯噪声")
    #椒盐噪声
    def jiaoYanNoise_event(self):
        print("椒盐噪声")
    #斑点噪声
    def splotNoise_event(self):
        print("斑点噪声")
    #泊松噪声
    def poissonNoise_event(self):
        print("泊松噪声")
    #end 噪声

    #start 变换

    #傅里叶变换
    def fourierTransform_event(self):
        print("傅里叶变换")
    #离散余弦变换
    def DCTTransformer_event(self):
        print("离散余弦变换")
    #Radom变换
    def RadomTransformer_event(self):
        print("Radom变换")

    #end 变换

    # start 编辑

    #放大图片
    def zoomUp_event(self):
        print("放大")
    #缩小图片
    def zoomDown_event(self):
        print("缩小")
    #灰度
    def grayLevel_event(self):
        print("灰度")
    #亮度调节
    def lightLevel_event(self):
        print("亮度")
    #旋转功能
    def rotate_event(self):
        print("旋转")
    #截图功能
    def printScreen_event(self):
        print("截图")
    #end 编辑
    # 显示图片
    def showPic_fun(self, path, zoomscale, desPosition):
        img = cv.imread(path)  # 读取图像
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # 转换图像通道
        x = img.shape[1]  # 获取图像大小
        y = img.shape[0]
        self.zoomscale = zoomscale  # 图片放缩尺度
        frame = QImage(img, x, y, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
        self.item.setScale(self.zoomscale)
        # desPosition.setGeometry(desPosition.pos().x(), desPosition.pos().x(), x*zoomscale, y*zoomscale)
        self.scene = QGraphicsScene()  # 创建场景
        self.scene.addItem(self.item)
        desPosition.setScene(self.scene)  # 将场景添加至视图

    def setHistory1_event(self):
        print("点击历史记录1")
        self.current_img = self.history1.text()
        self.current_idx = len(self.source_img_list)
        self.source_img_list.append(self.current_img)
        self.graphicsViewCenter.setHidden(False)
        print("类路径：",self.current_img)
        self.showPic_fun(self.current_img, 0.5, self.graphicsViewCenter)
    def setHistory2_event(self):
        self.current_img = self.history2.text()
        self.current_idx = len(self.source_img_list)
        self.source_img_list.append(self.current_img)
        self.graphicsViewCenter.setHidden(False)
        self.showPic_fun(self.current_img, 1, self.graphicsViewCenter)
    def setHistory3_event(self):
        self.current_img = self.history3.text()
        self.current_idx = len(self.source_img_list)
        self.source_img_list.append(self.current_img)
        self.graphicsViewCenter.setHidden(False)
        self.showPic_fun(self.current_img, 1, self.graphicsViewCenter)

    #打开图片
    def open_event(self):
        openfile_name = QFileDialog.getOpenFileName(self, '选择文件', 'D:/Picture',
                                                        'Image files(*.jpg , *.png)')
        if openfile_name[0]:
            self.imgCache_add(openfile_name[0])#将读取到的图片路径加入到缓存中
            self.current_img = openfile_name[0]
            self.current_idx = len(self.source_img_list)
            self.source_img_list.append(self.current_img)
            self.graphicsViewCenter.setHidden(False)
            self.showPic_fun(self.current_img,0.5,self.graphicsViewCenter)
        else:
            QMessageBox.warning(self,"提示","这个图片好像逃跑了！",QMessageBox.Yes)
    #保存图片
    def save_event(self):
        if  not self.current_img:
            QMessageBox.warning(self,"提示","什么都没有，我保存什么？",QMessageBox.Yes)
            return
        file_path = QFileDialog.getSaveFileName(self, '选择保存位置', 'D:/Picture/*.png',
                                                'Image files(*.png)')
        file_path = file_path[0]
        if file_path:
            cv.imwrite(file_path, cv.imread(self.current_img))

    def close_window(self):
        res = QMessageBox.warning(self, "退出系统", "是否确认退出", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if res == QMessageBox.Yes:
            app = QApplication.instance()
            # 退出应用程序
            app.quit()
    #打开图片后向缓存中加入一个记录
    def imgCache_add(self,path):
        f = open('../imgCache.txt', 'a+')
        f.write(path + '\n')
        f.close()
    #从已经读取过的图片缓存路径中读取最近使用的3条
    def imgCache_read(self):
        f = open('../imgCache.txt', 'r')
        lines = f.readlines()
        if len(lines)==0:
            print("缓存为空！")
            return
        position=0
        if len(lines) >= 3:#如果缓存中的历史记录超过三条
            for i in [-1,-2,-3]:
                self.cacheTxt.append(lines[i])
        else:
            for i in lines:
                self.cacheTxt.append(i)
        f.close()
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    mainWindon = MainWindow()
    mainWindon.show()

    sys.exit(app.exec_())
