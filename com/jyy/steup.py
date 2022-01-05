from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import *
import sys
import os
import cv2
from com.jyy.ui_python.UIModel import Ui_MainWindow
import cv2 as cv
import numpy as np
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
        self.graphicsViewRightTop.setHidden(True)
        self.graphicsViewRightCenter.setHidden(True)
        self.graphicsViewRightBottom.setHidden(True)
        self.graphicsViewCenter.setStyleSheet("border: 0px;background-color:#F0F0F0")#设置中间显示框的背景色，和边框
        self.graphicsViewLeft.setStyleSheet("border: 0px;background-color:#F0F0F0")  # 设置中间显示框的背景色，和边框
        self.graphicsViewRight.setStyleSheet("border: 0px;background-color:#F0F0F0")  # 设置中间显示框的背景色，和边框
        self.graphicsViewRightCenter.setStyleSheet("border: 0px;background-color:#F0F0F0")
        self.graphicsViewRightBottom.setStyleSheet("border: 0px;background-color:#F0F0F0")
        self.graphicsViewRightBottom.setStyleSheet("border: 0px;background-color:#F0F0F0")
        self.closeLightSlider(True)
        self.lightSlider.setMaximum(200)
        self.lightSlider.setMinimum(1)
        self.imgCache_read()#读取缓存路径
        self.zoomscale=1
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
        self.lightSlider.valueChanged['int'].connect(self.lightSlider_event)
        self.rotateSlider.valueChanged['int'].connect(self.rotate_event)
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
        img = cv2.imread(self.current_img.replace("\n",""))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        result = 20 * np.log(np.abs(fshift))
        self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_img, 0.5, self.graphicsViewLeft)
        self.graphicsViewRight.setHidden(False)#显示傅里叶变换后的图片
        x = result.shape[1]  # 获取图像大小
        y = result.shape[0]
        cv2.imwrite("cache/fuliye.jpg",result)
        result = cv2.imread("cache/fuliye.jpg")
        self.zoomscale = 0.5  # 图片放缩尺度
        frame = QImage(result, x, y, QImage.Format_Grayscale8)
        pix = QPixmap.fromImage(frame)
        self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
        self.item.setScale(self.zoomscale)
        self.scene = QGraphicsScene()  # 创建场景
        self.scene.addItem(self.item)
        self.graphicsViewRight.setScene(self.scene)  # 将场景添加至视图
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
        if (self.current_img == None):
            QMessageBox.warning(self, "提示", "您还没选择图片唉！", QMessageBox.Yes)
            return
        print("zoomscale:",self.zoomscale)
        if(self.zoomscale>=0.72):
            QMessageBox.warning(self, "提示", "图片够大了！", QMessageBox.Yes)
        else:
            self.showPic_fun(self.current_img,self.zoomscale+0.02,self.graphicsViewCenter)
    #缩小图片
    def zoomDown_event(self):
        print("缩小")
        if(self.current_img==None):
            QMessageBox.warning(self, "提示", "您还没选择图片唉！", QMessageBox.Yes)
            return
        if(self.zoomscale<=0.3):
            QMessageBox.warning(self, "提示", "够小了！", QMessageBox.Yes)
        else:
            self.showPic_fun(self.current_img, self.zoomscale - 0.02, self.graphicsViewCenter)
    #灰度
    def grayLevel_event(self):
        print("灰度")
        if(self.current_img==None):
            QMessageBox.warning(self, "提示", "还没有选择图片！", QMessageBox.Yes)
            return
        self.graphicsViewCenter.setHidden(True)#隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewRight.setHidden(False)
        self.showGrayPic_fun(self.current_img,0.5,self.graphicsViewRight)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_img, 0.5, self.graphicsViewLeft)
    #亮度调节 TODO
    def lightSlider_event(self):
        img = cv2.imread(self.current_img.replace('\n', ''))  # 读取图像
        x = img.shape[1]  # 获取图像大小
        y = img.shape[0]
        img_t = cv2.cvtColor(img, cv.COLOR_BGR2HSV)
        h, s, v = cv2.split(img_t)
        # 增加图像亮度
        v1 = np.clip(cv2.add(self.lightSlider.value()* v, 0), 0, 255)
        img = np.uint8(cv2.merge((h, s, v1)))
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)  # 转换图像通道
        self.zoomscale = self.zoomscale  # 图片放缩尺度
        frame = QImage(img, x, y, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
        self.item.setScale(self.zoomscale)
        self.scene = QGraphicsScene()  # 创建场景
        self.scene.addItem(self.item)
        self.graphicsViewCenter.setScene(self.scene)  # 将场景添加至视图
    def lightLevel_event(self):
        print("亮度")
    #旋转功能
    def rotate_event(self):
        print("旋转")
        img = cv2.imread(self.current_img.replace("\n",""))

        h, w = img.shape[:2]
        center = (w // 2, h // 2)

        # 旋转中心坐标，逆时针旋转：-90°，缩放因子：1
        M_2 = cv2.getRotationMatrix2D(center, self.rotateSlider.value(), 1)
        img = cv2.warpAffine(img, M_2, (w, h))
        # cv2.imshow("./rotated_-90.jpg", rotated_2)
        x = img.shape[1]  # 获取图像大小
        y = img.shape[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换图像通道
        self.zoomscale = 0.6  # 图片放缩尺度
        frame = QImage(img, x, y, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
        self.item.setScale(self.zoomscale)
        self.scene = QGraphicsScene()  # 创建场景
        self.scene.addItem(self.item)
        self.graphicsViewCenter.setScene(self.scene)  # 将场景添加至视图
    #截图功能 TODO
    def printScreen_event(self):
        print("截图")

    def RGB2HSI(rgb_img):
        """
        这是将RGB彩色图像转化为HSI图像的函数
        :param rgm_img: RGB彩色图像
        :return: HSI图像
        """
        # 保存原始图像的行列数
        row = np.shape(rgb_img)[0]
        col = np.shape(rgb_img)[1]
        # 对原始图像进行复制
        hsi_img = rgb_img.copy()
        # 对图像进行通道拆分
        B, G, R = cv2.split(rgb_img)
        # 把通道归一化到[0,1]
        [B, G, R] = [i / 255.0 for i in ([B, G, R])]
        H = np.zeros((row, col))  # 定义H通道
        I = (R + G + B) / 3.0  # 计算I通道
        S = np.zeros((row, col))  # 定义S通道
        for i in range(row):
            den = np.sqrt((R[i] - G[i]) ** 2 + (R[i] - B[i]) * (G[i] - B[i]))
            thetha = np.arccos(0.5 * (R[i] - B[i] + R[i] - G[i]) / den)  # 计算夹角
            h = np.zeros(col)  # 定义临时数组
            # den>0且G>=B的元素h赋值为thetha
            h[B[i] <= G[i]] = thetha[B[i] <= G[i]]
            # den>0且G<=B的元素h赋值为thetha
            h[G[i] < B[i]] = 2 * np.pi - thetha[G[i] < B[i]]
            # den<0的元素h赋值为0
            h[den == 0] = 0
            H[i] = h / (2 * np.pi)  # 弧度化后赋值给H通道
        # 计算S通道
        for i in range(row):
            min = []
            # 找出每组RGB值的最小值
            for j in range(col):
                arr = [B[i][j], G[i][j], R[i][j]]
                min.append(np.min(arr))
            min = np.array(min)
            # 计算S通道
            S[i] = 1 - min * 3 / (R[i] + B[i] + G[i])
            # I为0的值直接赋值0
            S[i][R[i] + B[i] + G[i] == 0] = 0
        # 扩充到255以方便显示，一般H分量在[0,2pi]之间，S和I在[0,1]之间
        hsi_img[:, :, 0] = H * 255
        hsi_img[:, :, 1] = S * 255
        hsi_img[:, :, 2] = I * 255
        return hsi_img

    def HSI2RGB(hsi_img):
        """
        这是将HSI图像转化为RGB图像的函数
        :param hsi_img: HSI彩色图像
        :return: RGB图像
        """
        # 保存原始图像的行列数
        row = np.shape(hsi_img)[0]
        col = np.shape(hsi_img)[1]
        # 对原始图像进行复制
        rgb_img = hsi_img.copy()
        # 对图像进行通道拆分
        H, S, I = cv2.split(hsi_img)
        # 把通道归一化到[0,1]
        [H, S, I] = [i / 255.0 for i in ([H, S, I])]
        R, G, B = H, S, I
        for i in range(row):
            h = H[i] * 2 * np.pi
            # H大于等于0小于120度时
            a1 = h >= 0
            a2 = h < 2 * np.pi / 3
            a = a1 & a2  # 第一种情况的花式索引
            tmp = np.cos(np.pi / 3 - h)
            b = I[i] * (1 - S[i])
            r = I[i] * (1 + S[i] * np.cos(h) / tmp)
            g = 3 * I[i] - r - b
            B[i][a] = b[a]
            R[i][a] = r[a]
            G[i][a] = g[a]
            # H大于等于120度小于240度
            a1 = h >= 2 * np.pi / 3
            a2 = h < 4 * np.pi / 3
            a = a1 & a2  # 第二种情况的花式索引
            tmp = np.cos(np.pi - h)
            r = I[i] * (1 - S[i])
            g = I[i] * (1 + S[i] * np.cos(h - 2 * np.pi / 3) / tmp)
            b = 3 * I[i] - r - g
            R[i][a] = r[a]
            G[i][a] = g[a]
            B[i][a] = b[a]
            # H大于等于240度小于360度
            a1 = h >= 4 * np.pi / 3
            a2 = h < 2 * np.pi
            a = a1 & a2  # 第三种情况的花式索引
            tmp = np.cos(5 * np.pi / 3 - h)
            g = I[i] * (1 - S[i])
            b = I[i] * (1 + S[i] * np.cos(h - 4 * np.pi / 3) / tmp)
            r = 3 * I[i] - g - b
            B[i][a] = b[a]
            G[i][a] = g[a]
            R[i][a] = r[a]
        rgb_img[:, :, 0] = B * 255
        rgb_img[:, :, 1] = G * 255
        rgb_img[:, :, 2] = R * 255
        return rgb_img
    #end 编辑
    #隐藏或显示亮度调节滑动条
    def closeLightSlider(self,flag):
        self.label.setHidden(flag)
        self.lightSlider.setHidden(flag)
        self.rotateThate.setHidden(flag)
        self.rotateSlider.setHidden(flag)
    #显示灰度图像
    def showGrayPic_fun(self, path, zoomscale, desPosition):
        img = cv2.imread(path.replace("\n",""))  # 读取图像
        x = img.shape[1]  # 获取图像大小
        y = img.shape[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换图像通道
        self.zoomscale = zoomscale  # 图片放缩尺度
        frame = QImage(img, x, y, QImage.Format_Grayscale8)
        pix = QPixmap.fromImage(frame)
        self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
        self.item.setScale(self.zoomscale)
        self.scene = QGraphicsScene()  # 创建场景
        self.scene.addItem(self.item)
        desPosition.setScene(self.scene)  # 将场景添加至视图
    # 显示图片
    def showPic_fun(self, path, zoomscale, desPosition):
        img = cv2.imread(path.replace('\n',''))  # 读取图像
        x = img.shape[1]  # 获取图像大小
        y = img.shape[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换图像通道
        self.zoomscale = zoomscale  # 图片放缩尺度
        frame = QImage(img, x, y, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item = QGraphicsPixmapItem(pix)  #创建像素图元
        self.item.setScale(self.zoomscale)
        self.scene = QGraphicsScene()  #创建场景
        self.scene.addItem(self.item)
        desPosition.setScene(self.scene)  #将场景添加至视图
    def showPicByImg_fun(self, img, zoomscale, desPosition):
        x = img.shape[1]  # 获取图像大小
        y = img.shape[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换图像通道
        self.zoomscale = zoomscale  # 图片放缩尺度
        frame = QImage(img, x, y, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item = QGraphicsPixmapItem(pix)  #创建像素图元
        self.item.setScale(self.zoomscale)
        self.scene = QGraphicsScene()  #创建场景
        self.scene.addItem(self.item)
        desPosition.setScene(self.scene)  #将场景添加至视图
    def setHistory1_event(self):
        self.current_img = self.history1.text().replace("\n","")
        self.current_idx = len(self.source_img_list)
        self.source_img_list.append(self.current_img)
        self.graphicsViewCenter.setHidden(False)
        self.graphicsViewLeft.setHidden(True)
        self.graphicsViewRight.setHidden(True)
        self.closeLightSlider(False)
        self.showPic_fun(self.current_img, 0.6, self.graphicsViewCenter)
    def setHistory2_event(self):
        self.current_img = self.history2.text().replace("\n","")
        self.current_idx = len(self.source_img_list)
        self.source_img_list.append(self.current_img)
        self.graphicsViewCenter.setHidden(False)
        self.graphicsViewLeft.setHidden(True)
        self.graphicsViewRight.setHidden(True)
        self.closeLightSlider(False)
        self.showPic_fun(self.current_img, 0.6, self.graphicsViewCenter)
    def setHistory3_event(self):
        self.current_img = self.history3.text().replace("\n","")
        self.current_idx = len(self.source_img_list)
        self.source_img_list.append(self.current_img)
        self.graphicsViewCenter.setHidden(False)
        self.graphicsViewLeft.setHidden(True)
        self.graphicsViewRight.setHidden(True)
        self.closeLightSlider(False)
        self.showPic_fun(self.current_img, 0.6, self.graphicsViewCenter)

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
            self.graphicsViewLeft.setHidden(True)
            self.graphicsViewRight.setHidden(True)
            self.closeLightSlider(False)
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
