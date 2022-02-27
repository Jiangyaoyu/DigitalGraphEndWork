from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import *
import sys
import os
import cv2

from com.jyy.test import evaluate_one_image
from com.jyy.ui_python.UIModel import Ui_MainWindow
import cv2 as cv
import numpy as np
import skimage
from PIL import Image,ImageDraw,ImageFont,ImageFilter
import traceback
import matplotlib.pyplot as plt
from screen_pic import ScreenShotsWin
T = 8
K = 8
channel = 3
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
        self.current_path = None #当前图片
        self.current_img=None
        self.right_path=None
        self.right_img= None
        self.current_idx = -1 #图片列表索引
        self.target_img = None #
        self.source_img_list = [] #图片列表
        self.cacheTxt = []  # 存放从缓存中读取的图片
        self.workCache= [] #存放在每次打开后移到左侧的所有图片 注意：存放的为读取后的图片数据，非图片路径
        self.workIdx = -1 #工作区图片缓存列表索引

        #初始化隐藏组件
        self.next.setHidden(True)
        self.back.setHidden(True)
        self.showCount.setHidden(True)
        self.graphicsViewLeft.setHidden(True)
        self.graphicsViewRight.setHidden(True)
        self.graphicsViewCenter.setHidden(True)
        self.label_2.setHidden(True)
        self.label_3.setHidden(True)
        self.graphicsViewCenter.setStyleSheet("border: 0px;background-color:#F0F0F0")#设置中间显示框的背景色，和边框
        self.graphicsViewLeft.setStyleSheet("border: 0px;background-color:#F0F0F0")  # 设置中间显示框的背景色，和边框
        self.graphicsViewRight.setStyleSheet("border: 0px;background-color:#F0F0F0")  # 设置中间显示框的背景色，和边框
        self.closeLightSlider(True)
        self.rightLeftButton.setHidden(True)
        self.leftRightButton.setHidden(True)
        self.lightSlider.setMinimum(5)
        self.lightSlider.setMaximum(15)
        self.lightSlider.setSingleStep(0)
        self.lightSlider.setPageStep(1)
        self.lightSlider.setProperty("value", 10)
        self.imgCache_read()#读取缓存路径
        self.zoomscale=1
        print("初始化时历史记录个数：", len(self.cacheTxt), len(self.workCache))
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
        self.rotate.triggered.connect(self.rotate90_event)
        self.lightLevel.triggered.connect(self.lightLevel_event)
        self.grayLevel.triggered.connect(self.grayLevel_event)
        self.zoomDown.triggered.connect(self.zoomDown_event)
        self.zoomUp.triggered.connect(self.zoomUp_event)
        self.HSVColorModel.triggered.connect(self.HSVColorModel_event)
        self.YCbCrColorModel.triggered.connect(self.YCbCrColorModel_event)
        self.NTSCColorModel.triggered.connect(self.NTSCColorModel_event)
        self.histogramBalance.triggered.connect(self.histogramBalance_event)
        self.colorReinforce.triggered.connect(self.colorReinforce_event)
        self.noColorRenforce.triggered.connect(self.noColorRenforce_event)
        self.BHistgram.triggered.connect(self.BHistgram_event)
        self.GHistogram.triggered.connect(self.GHistogram_event)
        self.RHistogram.triggered.connect(self.RHistogram_event)
        self.sharpenFilterNoLinear.triggered.connect(self.sharpenFilterNoLinear_event)
        self.sharpenFilterLinear.triggered.connect(self.sharpenFilterLinear_event)
        self.smoothFilterNoLinear.triggered.connect(self.smoothFilterNoLinear_event)
        self.smoothFilterLinear.triggered.connect(self.smoothFilterLinear_event)
        self.lowPassFilter.triggered.connect(self.lowPassFilter_event)
        self.highPassFilter.triggered.connect(self.highPassFilter_event)
        self.next.clicked.connect(self.next_event)
        self.back.clicked.connect(self.back_event)
        self.history1.triggered.connect(self.setHistory1_event)
        self.history2.triggered.connect(self.setHistory2_event)
        self.history3.triggered.connect(self.setHistory3_event)
        self.lightSlider.valueChanged['int'].connect(self.lightSlider_event)
        self.rotateSlider.valueChanged['int'].connect(self.rotate_event)
        self.closeProcess.triggered.connect(self.closeProces_event)
        self.openProcess.triggered.connect(self.openProcess_event)
        self.FeatureExtraction.triggered.connect(self.FeatuerExtraction_event)
        self.ImgClassfiyAndRcong.triggered.connect(self.ImgClassfiyAndRcong_event)
        self.thresholdValue.triggered.connect(self.thresholdValue_event)
        self.leftRightButton.clicked.connect(self.leftRightButton_event)
        self.rightLeftButton.clicked.connect(self.rightLeftButton_event)
        self.addTextMark.triggered.connect(self.addTextMark_event)
        self.dilate.triggered.connect(self.dilate_event)
        self.dingmao.triggered.connect(self.dingmao_event)
        self.erode.triggered.connect(self.erode_event)
        self.xingtaiixuetidu.triggered.connect(self.xingtaiixuetidu_event)
        self.heimao.triggered.connect(self.heimao_event)

    def dilate_event(self):#膨胀
        print("dilate_event")
        img = cv2.imread(self.current_path, 1)
        # 结构元素
        kernel = cv2.getStructuringElement(cv.MORPH_RECT, (3, 3))
        # 膨胀图像
        res = cv2.dilate(img, kernel)
        self.saveVariable(self.current_path, '', '', res)
        self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewRight.setHidden(False)
        self.showPicByImg_fun(res, 1, self.graphicsViewRight)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)
    def dingmao_event(self):
        print("dingmao_event")
        img = cv2.imread(self.current_path, 1)
        H, W, C = img.shape
        # # Otsu binary
        # # Grayscale
        out = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
        out = out.astype(np.uint8)
        # Determine threshold of Otsu's binarization
        max_sigma = 0
        max_t = 0
        for _t in range(1, 255):
            v0 = out[np.where(out < _t)]
            m0 = np.mean(v0) if len(v0) > 0 else 0.
            w0 = len(v0) / (H * W)
            v1 = out[np.where(out >= _t)]
            m1 = np.mean(v1) if len(v1) > 0 else 0.
            w1 = len(v1) / (H * W)
            sigma = w0 * w1 * ((m0 - m1) ** 2)
            if sigma > max_sigma:
                max_sigma = sigma
                max_t = _t
        # Binarization
        th = max_t
        out[out < th] = 0
        out[out >= th] = 255
        # 设置卷积核
        kernel = np.ones((3, 3), np.uint8)
        # 顶帽运算
        dst = cv2.morphologyEx(out, cv2.MORPH_TOPHAT, kernel)
        self.saveVariable(self.current_path, '', '', dst)
        self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewRight.setHidden(False)
        self.showPicByImg_fun(dst, 1, self.graphicsViewRight)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)
    def erode_event(self):#腐蚀
        print("erode_event腐蚀")
        img = cv2.imread(self.current_path, 1)
        # 结构元素
        kernel = cv2.getStructuringElement(cv.MORPH_RECT, (3, 3))
        # 腐蚀图像
        res = cv2.erode(img, kernel)
        self.saveVariable(self.current_path, '', '', res)
        self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewRight.setHidden(False)
        self.showPicByImg_fun(res, 1, self.graphicsViewRight)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)
    def xingtaiixuetidu_event(self):
        print("xingtaiixuetidu_event")
        img = cv2.imread(self.current_path, 1)
        kernel = np.ones((3, 3), np.uint8)
        img = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
        self.saveVariable(self.current_path, '', '', img)
        self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewRight.setHidden(False)
        self.showPicByImg_fun(img, 1, self.graphicsViewRight)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)
    def heimao_event(self):
        print("heimao_event")
        img = cv2.imread(self.current_path, 1)
        H, W, C = img.shape
        # Otsu binary
        # Grayscale
        out = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
        out = out.astype(np.uint8)
        # Determine threshold of Otsu's binarization
        max_sigma = 0
        max_t = 0
        for _t in range(1, 255):
            v0 = out[np.where(out < _t)]
            m0 = np.mean(v0) if len(v0) > 0 else 0.
            w0 = len(v0) / (H * W)
            v1 = out[np.where(out >= _t)]
            m1 = np.mean(v1) if len(v1) > 0 else 0.
            w1 = len(v1) / (H * W)
            sigma = w0 * w1 * ((m0 - m1) ** 2)
            if sigma > max_sigma:
                max_sigma = sigma
                max_t = _t
        # Binarization
        th = max_t
        out[out < th] = 0
        out[out >= th] = 255
        # 设置卷积核
        kernel = np.ones((3, 3), np.uint8)
        # 黑帽运算
        dst = cv2.morphologyEx(out, cv2.MORPH_BLACKHAT, kernel)
        self.saveVariable(self.current_path, '', '', dst)
        self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewRight.setHidden(False)
        self.showPicByImg_fun(dst, 1, self.graphicsViewRight)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)
    def next_event(self):
        print("next_event")
        print("debug")
        print(self.workIdx)
        if(self.workIdx<len(self.workCache)-1):
            self.current_path = self.workCache[self.workIdx+1]
            self.showCount.setText('{}/{}'.format(self.workIdx+1,len(self.workCache)))
            self.workIdx += 1
        else:
            print("next_else",self.workIdx,len(self.workCache))
            self.current_path = self.workCache[self.workIdx]
            print(self.current_path)
            self.showCount.setText('{}/{}'.format(self.workIdx+1, len(self.workCache)))
        self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)
    def back_event(self):
        print("back_event")
        print("debug")
        print(self.workIdx)
        if (self.workIdx >0):
            self.current_path = self.workCache[self.workIdx-1]
            self.showCount.setText('{}/{}'.format(self.workIdx -1, len(self.workCache)))
            self.workIdx -=1
        else:

            self.current_path = self.workCache[self.workIdx]
            self.showCount.setText('{}/{}'.format(self.workIdx+1, len(self.workCache)))
        self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)
    def leftRightButton_event(self):
        print("leftRightButton_event")
        # self.graphicsViewLeft.setHidden(True)
        self.right_path=self.current_path
        if self.workIdx>0:
            self.workIdx -= 1
            self.workCache.remove(self.workCache[self.workIdx+1])
            self.showCount.setText('{}/{}'.format(self.workIdx + 1, len(self.workCache)))
            self.current_path = self.workCache[self.workIdx]
        else:
            return
        #self.current_path=''
        self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewRight.setHidden(False)
        self.showPic_fun(self.right_path, 1, self.graphicsViewRight)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)
    def rightLeftButton_event(self):
        print("rightLeftButton_event")
        print('self_right_path:', self.right_path, 'self_current_path:', self.current_path)
        # self.graphicsViewRight.setHidden(True)
        if self.right_path=='' and self.right_img!='':
            self.current_idx += 1
            self.workIdx +=1
            self.showCount.setText('{}/{}'.format(self.workIdx +1, len(self.workCache)))
            cv2.imwrite("./cache/tranformer_{}.jpg".format(self.current_idx),self.right_img)
            self.current_path="./cache/tranformer_{}.jpg".format(self.current_idx)
            self.workCache.append(self.current_path)  # 加入工作区缓存
            self.current_img = self.right_img
            self.right_img=''
            self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
            self.closeLightSlider(True)
            self.graphicsViewRight.setHidden(False)
            self.showPicByImg_fun(self.right_img, 1, self.graphicsViewRight)
            self.graphicsViewLeft.setHidden(False)
            self.showPicByImg_fun(self.current_img, 1, self.graphicsViewLeft)
        if self.right_img=='' and self.right_path!='':
            self.current_idx += 1
            self.workIdx += 1
            self.showCount.setText('{}/{}'.format(self.workIdx + 1, len(self.workCache)))
            img = cv2.imread(self.right_path)
            cv2.imwrite("./cache/tranformer_{}.jpg".format(self.current_idx), img)
            self.workCache.append("./cache/tranformer_{}.jpg".format(self.current_idx))#加入工作区缓存
            self.current_path = self.right_path
            self.right_path=''
            self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
            self.closeLightSlider(True)
            #self.graphicsViewRight.setHidden(False)
            #self.showPic_fun(self.right_path, 1, self.graphicsViewRight)
            self.graphicsViewLeft.setHidden(False)
            self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)
        else :
            return
    def saveVariable(self,current_path,current_img,right_path,right_img):
        self.current_path=current_path
        self.current_img=current_img
        self.right_path=right_path
        self.right_img=right_img
    #添加水印
    def addTextMark_event(self):
        print("addTextMark_event")
        im_before = Image.open(self.current_path)
        draw = ImageDraw.Draw(im_before)
        myfont = ImageFont.truetype('C:/windows/fonts/simhei.ttf', size=50)
        fillcolor = "#ff0000"
        value, ok = QInputDialog.getText(self, '水印', '请输入添加的水印', QLineEdit.Normal, '添加水印')
        draw.text((100, 8), value, font=myfont, fill=fillcolor)
        im_before.save('./cache/addTextMark.png')
        self.saveVariable(self.current_path,'','./cache/addTextMark.png','')
        self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewRight.setHidden(False)
        self.showPic_fun('./cache/addTextMark.png', 1, self.graphicsViewRight)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)
    #形态学处理
    def closeProces_event(self):
        print("closeProces_event")
        img = cv2.imread(self.current_path, 1)
        # OpenCV定义的结构元素
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        self.saveVariable(self.current_path,'','',closed)
        self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewRight.setHidden(False)
        self.showPicByImg_fun(closed, 1, self.graphicsViewRight)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)
    def openProcess_event(self):
        print("openProcess_event")
        img = cv2.imread(self.current_path, 1)
        # OpenCV定义的结构元素
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
         # 开运算
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        self.saveVariable(self.current_path,'','',opened)
        self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewRight.setHidden(False)
        self.showPicByImg_fun(opened, 1, self.graphicsViewRight)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)
        # 显示腐蚀后的图像
    #特征提取
    def FeatuerExtraction_event(self):
        print("FeatuerExtraction_event")
        img = cv2.imread(self.current_path)
        # kaze检测
        kaze = cv2.KAZE_create()
        keypoints = kaze.detect(img, None)
        img1 = img.copy()
        kaze_img = cv2.drawKeypoints(img, keypoints, img1, color=(0, 255, 0))
        self.saveVariable(self.current_path,'','',kaze_img)
        self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewRight.setHidden(False)
        self.showPicByImg_fun(kaze_img, 1, self.graphicsViewRight)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)
    #图像分类与识别
    def ImgClassfiyAndRcong_event(self):
        print("ImgClassfiyAndRcong_event")
        img = Image.open(self.current_path)
        imag = img.resize([100, 100])
        image = np.array(imag)
        text = evaluate_one_image(image)
        QMessageBox.warning(self, "识别结果", text, QMessageBox.Yes)
    #阈值分割
    def thresholdValue_event(self):
        print("阈值分割")
        img = cv2.imread(self.current_path, 0)  # 0是第二个参数，将其转为灰度图
        value, ok = QInputDialog.getText(self, "阈值分割", "输入阈值大小:", QLineEdit.Normal, "130")

        c=np.int(value.replace(' ', ''))
        # 利用cv2.threshhold()函数进行简单阈值分割，第一个参数是待分割图像，第二个参数是阈值大小
        # 第三个参数是赋值的像素值，第四个参数是阈值分割方法
        ret, thresh1 = cv2.threshold(img, c, 255, cv2.THRESH_BINARY)
        self.saveVariable(self.current_path,'','',thresh1)
        self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewRight.setHidden(False)
        self.showPicByImg_fun(thresh1, 1, self.graphicsViewRight)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)
    #start 色彩增强
    def noColorRenforce_event(self):
        print("伪色彩图像")
        im_gray = cv2.imread(self.current_path, cv2.IMREAD_GRAYSCALE)
        im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)
        self.saveVariable(self.current_path,'','',im_color)
        self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewRight.setHidden(False)
        self.showPicByImg_fun(im_color, 1, self.graphicsViewRight)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)
    def colorReinforce_event(self):
        print("真色彩图像")
        # RGB到HSI的变换
        def rgb2hsi(image):
            b, g, r = cv.split(image)  # 读取通道
            r = r / 255.0  # 归一化
            g = g / 255.0
            b = b / 255.0
            eps = 1e-6  # 防止除零

            img_i = (r + g + b) / 3  # I分量

            img_h = np.zeros(r.shape, dtype=np.float32)
            img_s = np.zeros(r.shape, dtype=np.float32)
            min_rgb = np.zeros(r.shape, dtype=np.float32)
            # 获取RGB中最小值
            min_rgb = np.where((r <= g) & (r <= b), r, min_rgb)
            min_rgb = np.where((g <= r) & (g <= b), g, min_rgb)
            min_rgb = np.where((b <= g) & (b <= r), b, min_rgb)
            img_s = 1 - 3 * min_rgb / (r + g + b + eps)  # S分量

            num = ((r - g) + (r - b)) / 2
            den = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
            theta = np.arccos(num / (den + eps))
            img_h = np.where((b - g) > 0, 2 * np.pi - theta, theta)  # H分量
            img_h = np.where(img_s == 0, 0, img_h)

            img_h = img_h / (2 * np.pi)  # 归一化
            temp_s = img_s - np.min(img_s)
            temp_i = img_i - np.min(img_i)
            img_s = temp_s / np.max(temp_s)
            img_i = temp_i / np.max(temp_i)

            image_hsi = cv.merge([img_h, img_s, img_i])
            # return img_h, img_s, img_i, image_hsi
            return image_hsi

        # HSI到RGB的变换
        def hsi2rgb(image_hsi):
            eps = 1e-6
            img_h, img_s, img_i = cv.split(image_hsi)

            image_out = np.zeros((img_h.shape[0], img_h.shape[1], 3))
            img_h = img_h * 2 * np.pi
            # print(img_h)

            img_r = np.zeros(img_h.shape, dtype=np.float32)
            img_g = np.zeros(img_h.shape, dtype=np.float32)
            img_b = np.zeros(img_h.shape, dtype=np.float32)

            # 扇区1
            img_b = np.where((img_h >= 0) & (img_h < 2 * np.pi / 3), img_i * (1 - img_s), img_b)
            img_r = np.where((img_h >= 0) & (img_h < 2 * np.pi / 3),
                             img_i * (1 + img_s * np.cos(img_h) / (np.cos(np.pi / 3 - img_h))), img_r)
            img_g = np.where((img_h >= 0) & (img_h < 2 * np.pi / 3), 3 * img_i - (img_r + img_b), img_g)

            # 扇区2                                                                                        # H=H-120°
            img_r = np.where((img_h >= 2 * np.pi / 3) & (img_h < 4 * np.pi / 3), img_i * (1 - img_s), img_r)
            img_g = np.where((img_h >= 2 * np.pi / 3) & (img_h < 4 * np.pi / 3),
                             img_i * (1 + img_s * np.cos(img_h - 2 * np.pi / 3) / (np.cos(np.pi - img_h))), img_g)
            img_b = np.where((img_h >= 2 * np.pi / 3) & (img_h < 4 * np.pi / 3), 3 * img_i - (img_r + img_g), img_b)

            # 扇区3                                                                                        # H=H-240°
            img_g = np.where((img_h >= 4 * np.pi / 3) & (img_h <= 2 * np.pi), img_i * (1 - img_s), img_g)
            img_b = np.where((img_h >= 4 * np.pi / 3) & (img_h <= 2 * np.pi),
                             img_i * (1 + img_s * np.cos(img_h - 4 * np.pi / 3) / (np.cos(5 * np.pi / 3 - img_h))),
                             img_b)
            img_r = np.where((img_h >= 4 * np.pi / 3) & (img_h <= 2 * np.pi), 3 * img_i - (img_b + img_g), img_r)

            temp_r = img_r - np.min(img_r)
            img_r = temp_r / np.max(temp_r)

            temp_g = img_g - np.min(img_g)
            img_g = temp_g / np.max(temp_g)

            temp_b = img_b - np.min(img_b)
            img_b = temp_b / np.max(temp_b)

            image_out = cv.merge((img_b, img_g, img_r))  # 按RGB合并，后面不用转换通道
            # print(image_out.shape)
            return image_out

        img = cv2.imread(self.current_path, cv2.IMREAD_UNCHANGED)
        hsi = rgb2hsi(img)
        hsi[:, :, 1] = hsi[:, :, 1] * 1.1
        hsi[:, :, 2] = hsi[:, :, 2] * 1.8
        out = hsi2rgb(hsi)

        out = (out * 255).astype(np.uint8)
        self.saveVariable(self.current_path,'','',out)
        self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewRight.setHidden(False)
        self.showPicByImg_fun(out, 1, self.graphicsViewRight)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)
    def histogramBalance_event(self):
        print("直方图均衡化")
        img = cv2.imread(self.current_path, cv2.IMREAD_UNCHANGED)
        # 通道分解
        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]

        bH = cv.equalizeHist(b)
        gH = cv.equalizeHist(g)
        rH = cv.equalizeHist(r)

        # 通道合成
        img[:, :, 0] = bH
        img[:, :, 1] = gH
        img[:, :, 2] = rH

        self.saveVariable(self.current_path,'','',img)
        self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewRight.setHidden(False)
        self.showPicByImg_fun(img, 1, self.graphicsViewRight)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)
    def NTSCColorModel_event(self):
        print("NTSC颜色模型")
        img = cv2.imread(self.current_path, cv2.IMREAD_UNCHANGED)

        img_rows = int(img.shape[0])
        img_cols = int(img.shape[1])
        yiq_image = img.copy()
        R, G, B = cv2.split(yiq_image)

        for x in range(img_rows):
            for y in range(img_cols):
                right_matrix = np.array([[R[x, y]],
                                         [G[x, y]],
                                         [B[x, y]]])
                left_matrix = np.array([[0.299, 0.587, 0.114],
                                        [0.596, -0.275, -0.321],
                                        [0.212, -0.528, 0.311]])
                matrix = np.dot(left_matrix, right_matrix)
                r = matrix[0][0]
                g = matrix[1][0]
                b = matrix[2][0]
                yiq_image[x, y] = (r, g, b)

        '''-----------------YIQ → RGB------------------------'''
        img_row = int(yiq_image.shape[0])
        img_col = int(yiq_image.shape[1])
        rgb_image = yiq_image.copy()
        Y, I, Q = cv.split(rgb_image)
        for x in range(img_row):
            for y in range(img_col):
                right_matrix = np.array([[Y[x, y]],
                                         [I[x, y]],
                                         [Q[x, y]]])
                left_matrix = np.array([[1, 0.956, 0.620],
                                        [1, -0.272, -0.647],
                                        [1, -1.108, 1.705]])
                matrix = np.dot(left_matrix, right_matrix)
                r = matrix[0][0]
                g = matrix[1][0]
                b = matrix[2][0]
                rgb_image[x, y] = (r, g, b)
        self.saveVariable(self.current_path,'','',rgb_image)
        self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewRight.setHidden(False)
        self.showPicByImg_fun(rgb_image, 1, self.graphicsViewRight)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)

    def gamma_enhance(self, mat, gamma=0.9):
        tar = mat.copy()
        tar = tar * 1.0 / 255
        tar = np.power(tar, gamma)
        tar = (tar * 255).astype(np.uint8)
        return tar
    def YCbCrColorModel_event(self):
        print("YCBCR颜色模型")
        img = cv2.imread(self.current_path, cv2.IMREAD_UNCHANGED)
        gamma = 0.8
        res = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        res[:, :, 0] = self.gamma_enhance(res[:, :, 0], gamma)
        res = cv2.cvtColor(res, cv2.COLOR_YCR_CB2BGR)
        cv2.imwrite("cache/ycbcr.jpg", res)
        self.saveVariable(self.current_path, '', 'cache/ycbcr.jpg', '')
        self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewRight.setHidden(False)
        self.showPic_fun("cache/ycbcr.jpg", 1, self.graphicsViewRight)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)
    def HSVColorModel_event(self):
        print("HSVC颜色模型")
        img = cv2.imread(self.current_path, cv2.IMREAD_UNCHANGED)
        gamma = 1.8

        res = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        res[:, :, 1] = self.gamma_enhance(res[:, :, 1], gamma)
        res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
        cv2.imwrite("cache/hsv.jpg",res)
        self.saveVariable(self.current_path, '', "cache/hsv.jpg", '')
        self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewRight.setHidden(False)
        self.showPic_fun("cache/hsv.jpg", 1, self.graphicsViewRight)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)
    #end 色彩增强

    #start 直方图
    def RHistogram_event(self):  # R直方图
        print("R")
        im1 = Image.open(self.current_path)
        # 将RGB三个通道分开
        r, g, b = im1.split()
        ar = np.array(r).flatten()
        plt.cla()
        plt.hist(ar, bins=256, color='gray')
        plt.savefig('./cache/R.png', dpi=65)
        self.saveVariable(self.current_path, '', './cache/R.png', '')
        self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewRight.setHidden(False)
        self.showPic_fun('./cache/R.png', 1, self.graphicsViewRight)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)

    def GHistogram_event(self):  # G直方图
        print("G")
        im1 = Image.open(self.current_path)
        # 将RGB三个通道分开
        r, g, b = im1.split()
        ag = np.array(g).flatten()
        plt.cla()
        plt.hist(ag, bins=256, color='gray')
        plt.savefig('./cache/G.png', dpi=65)
        self.saveVariable(self.current_path, '', './cache/G.png', '')
        self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewRight.setHidden(False)
        self.showPic_fun('./cache/G.png', 1, self.graphicsViewRight)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)

    def BHistgram_event(self):  # B直方图
        print("B")
        im1 = Image.open(self.current_path)
        # 将RGB三个通道分开
        r, g, b = im1.split()
        ab = np.array(b).flatten()
        plt.cla()
        plt.hist(ab, bins=256,color='gray')
        plt.savefig('./cache/B.png', dpi=65)
        self.saveVariable(self.current_path, '', './cache/B.png', '')
        self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewRight.setHidden(False)
        self.showPic_fun('./cache/B.png', 1, self.graphicsViewRight)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)

    #end 直方图
    #start 滤波
    def highPassFilter_event(self):
        print("高通滤波")
        # 以灰度的方式加载图片
        img = cv2.imread(self.current_path, cv2.IMREAD_UNCHANGED)
        img = np.mean(img, axis=2)
        # img = self.toGray(img)

        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)

        # def highPassFiltering(img, size):  # 传递参数为傅里叶变换后的频谱图和滤波尺寸
        #     h, w = img.shape[0:2]  # 获取图像属性
        #     h1, w1 = int(h / 2), int(w / 2)  # 找到傅里叶频谱图的中心点
        #     img[h1 - int(size / 2):h1 + int(size / 2), w1 - int(size / 2):w1 + int(size / 2)] = -1
        #     img[h1, w1] = 8
        #     # 中心点加减滤波尺寸的一半，刚好形成一个定义尺寸的滤波大小，然后设置为0
        #     return img
        #
        # # 调用高通滤波函数
        # img1 = highPassFiltering(fshift, 80)

        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        fshift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
        # 傅里叶逆变换
        ishift = np.fft.ifftshift(fshift)
        iimg = np.fft.ifft2(ishift)
        iimg = np.abs(iimg)
        cv2.imwrite("cache/gaotong.jpg",iimg)
        self.saveVariable(self.current_path, '', 'cache/gaotong.jpg', '')
        self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewRight.setHidden(False)
        self.showPic_fun('cache/gaotong.jpg', 1, self.graphicsViewRight)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)
    def lowPassFilter_event(self):
        print("低通滤波")
        img = cv2.imread(self.current_path, cv2.IMREAD_UNCHANGED)
        img = np.mean(img, axis=2)

        # 傅里叶变换
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        fshift = np.fft.fftshift(dft)

        # 设置低通滤波器
        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置
        mask = np.zeros((rows, cols, 2), np.uint8)
        mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1

        # 掩膜图像和频谱图像乘积
        f = fshift * mask
        # 傅里叶逆变换
        ishift = np.fft.ifftshift(f)
        iimg = cv2.idft(ishift)
        res = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])

        res = np.interp(res, (res.min(), res.max()), (0, 255))
        res = np.uint8(res)

        self.saveVariable(self.current_path, '', '', res)
        self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewRight.setHidden(False)
        self.showPicByImg_fun(res, 1, self.graphicsViewRight)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)
    def smoothFilterNoLinear_event(self):
        print("平滑滤波——非线性")
        img = cv2.imread(self.current_path, cv2.IMREAD_ANYCOLOR)
        out = cv2.medianBlur(img, 3)  # 中值滤波函数
        self.saveVariable(self.current_path, '', '', out)
        self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewRight.setHidden(False)
        self.showPicByImg_fun(out, 1, self.graphicsViewRight)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)
    def smoothFilterLinear_event(self):
        print("平滑滤波-线性")
        img = cv2.imread(self.current_path,cv2.IMREAD_UNCHANGED)
        kernel = np.ones((3, 3), np.float32) / 9
        out = cv2.filter2D(img, -1, kernel)  # ddepth=-1 表示输出和原图像深度（通道数）相同
        self.saveVariable(self.current_path, '', '', out)
        self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewRight.setHidden(False)
        self.showPicByImg_fun(out, 1, self.graphicsViewRight)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)
    def sharpenFilterNoLinear_event(self):
        print("锐化滤波-非线性")
        img = cv2.imread(self.current_path, cv2.IMREAD_ANYCOLOR)
        kernel = np.float32([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        out = cv2.filter2D(img, -1, kernel)
        self.saveVariable(self.current_path, '', '', out)
        self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewRight.setHidden(False)
        self.showPicByImg_fun(out, 1, self.graphicsViewRight)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)
    def sharpenFilterLinear_event(self):
        print("锐化滤波-线性")
        # 锐化滤波
        img = cv2.imread(self.current_path,cv2.IMREAD_UNCHANGED)
        kernel = np.float32([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        out = cv2.filter2D(img, -1, kernel)
        self.saveVariable(self.current_path, '', '', out)
        self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewRight.setHidden(False)
        self.showPicByImg_fun(out, 1, self.graphicsViewRight)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)
    #end 滤波


    #start 噪声

    #高斯噪声
    def clamp(pv):
        """防止溢出"""
        if pv > 255:
            return 255
        elif pv < 0:
            return 0
        else:
            return pv
    def GaussNoise_event(self):
        print("高斯噪声")

        def _gasuss_noise(image, mean=0.0, var=0.1):
            copyImage = image.copy()
            noise = np.random.normal(loc=mean, scale=var, size=copyImage.shape)
            copyImage = np.array(copyImage / 255, dtype=float)
            out = copyImage + noise
            out = np.clip(out, 0.0, 1.0)
            out = np.uint8(out * 255)
            return out
        img = cv2.imread(self.current_path, cv.IMREAD_UNCHANGED)
        blur = _gasuss_noise(image=img)
        cv2.imwrite('./cache/gaussian_noise_img.jpg', blur)
        #self.target_img = self.cv2file(blur)
        # srcImage = cv2.imread(self.current_path)
        # b, g, r = cv2.split(srcImage)  # 先将bgr格式拆分
        # srcimg = cv2.merge([r, g, b])
        # gaussian_noise_img = skimage.util.random_noise(srcimg, mode='gaussian')*255
        # cv2.imwrite('./cache/gaussian_noise_img.jpg', gaussian_noise_img)
        self.saveVariable(self.current_path, '', './cache/gaussian_noise_img.jpg', '')
        self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewRight.setHidden(False)
        self.showPic_fun("./cache/gaussian_noise_img.jpg", 1, self.graphicsViewRight)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)
    #椒盐噪声
    def jiaoYanNoise_event(self):
        print("椒盐噪声")
        img = cv2.imread(self.current_path,cv.IMREAD_UNCHANGED)
        # b, g, r = cv2.split(image)  # 先将bgr格式拆分
        # img = cv2.merge([r, g, b])
        # m = int((img.shape[0] * img.shape[1]) * 0.01)
        # for a in range(m):
        #     i = int(np.random.random() * img.shape[1])
        #     j = int(np.random.random() * img.shape[0])
        #     if img.ndim == 2:
        #         img[j, i] = 255
        #     elif img.ndim == 3:
        #         img[j, i, 0] = 255
        #         img[j, i, 1] = 255
        #         img[j, i, 2] = 255
        # for b in range(m):
        #     i = int(np.random.random() * img.shape[1])
        #     j = int(np.random.random() * img.shape[0])
        #     if img.ndim == 2:
        #         img[j, i] = 0
        #     elif img.ndim == 3:
        #         img[j, i, 0] = 0
        #         img[j, i, 1] = 0
        #         img[j, i, 2] = 0
        # 信噪比
        SNR = 0.6

        # 计算总像素数目 SP， 得到要加噪的像素数目 NP = SP * (1-SNR)
        noiseNum = int((1 - SNR) * img.shape[0] * img.shape[1])

        # 于随机位置将像素值随机指定为0或者255
        for i in range(noiseNum):

            randX = np.random.random_integers(0, img.shape[0] - 1)

            randY = np.random.random_integers(0, img.shape[1] - 1)

            if np.random.random_integers(0, 1) == 0:

                img[randX, randY] = 0

            else:

                img[randX, randY] = 255
        self.saveVariable(self.current_path, '', '', img)
        self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewRight.setHidden(False)
        self.showPicByImg_fun(img, 1, self.graphicsViewRight)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)
    #斑点噪声
    def splotNoise_event(self):
        print("斑点噪声")
        img = cv2.imread(self.current_path,cv.IMREAD_UNCHANGED)
        def _speckle_noise(image, mean=0.0, var=0.2):
            copyImage = image.copy()
            noise = np.random.normal(loc=mean, scale=var, size=copyImage.shape)
            copyImage = np.array(copyImage / 255, dtype=float)
            out = (1 + noise) * copyImage
            out = np.clip(out, 0.0, 1.0)
            out = np.uint8(out * 255)
            return out
        blur = _speckle_noise(image=img)
        # b, g, r = cv2.split(image)  # 先将bgr格式拆分
        # img = cv2.merge([r, g, b])
        # speckle_noise_img = skimage.util.random_noise(img, mode='speckle')*255
        cv2.imwrite('./cache/speckle_noise_img.jpg', blur)
        self.saveVariable(self.current_path, '', './cache/speckle_noise_img.jpg', '')
        self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewRight.setHidden(False)
        self.showPic_fun('./cache/speckle_noise_img.jpg', 1, self.graphicsViewRight)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)
    #泊松噪声
    def poissonNoise_event(self):
        print("泊松噪声")
        img = cv2.imread(self.current_path,cv.IMREAD_UNCHANGED)
        img = img.astype(float)
        # noisy image
        noise_mask = np.random.poisson(img / 255.0 * 0.8) / 0.8 * 255
        # noise_mask = np.random.poisson(img)

        noisy_img = img + noise_mask
        noisy_img = noisy_img.astype(np.uint8)
        # b, g, r = cv2.split(image)  # 先将bgr格式拆分
        # img = cv2.merge([r, g, b])
        # poisson_noise_img = skimage.util.random_noise(img, mode='poisson', seed=None, clip=True)*255
        cv2.imwrite('./cache/poisson_noise_img.jpg', noisy_img)
        self.saveVariable(self.current_path, '', './cache/poisson_noise_img.jpg', '')
        self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewRight.setHidden(False)
        self.showPic_fun('./cache/poisson_noise_img.jpg', 1, self.graphicsViewRight)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)
    #end 噪声

    #start 变换

    #傅里叶变换
    def fourierTransform_event(self):
        print("傅里叶变换")
        img = cv2.imread(self.current_path.replace("\n",""))
        img = np.mean(img[..., :min(3, img.shape[-1])], axis=2)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)  # 得到结果为复数
        magnitude_spectrum = 10 * np.log(np.abs(fshift))  # 先取绝对值，表示取模。取对数，将数据范围变小
        self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)
        self.graphicsViewRight.setHidden(False)#显示傅里叶变换后的图片
        x = magnitude_spectrum.shape[1]  # 获取图像大小
        y = magnitude_spectrum.shape[0]
        cv2.imwrite("cache/fuliye.jpg",magnitude_spectrum)
        result = cv2.imread("cache/fuliye.jpg")
        self.saveVariable(self.current_path, '', "cache/fuliye.jpg", '')
        self.showPic_fun("cache/fuliye.jpg",1,self.graphicsViewRight)
        # self.zoomscale = 1  # 图片放缩尺度
        # frame = QImage(result, x, y, QImage.Format_Grayscale8)
        # pix = QPixmap.fromImage(frame)
        # self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
        # self.item.setScale(self.zoomscale)
        # self.scene = QGraphicsScene()  # 创建场景
        # self.scene.addItem(self.item)
        # self.graphicsViewRight.setScene(self.scene)  # 将场景添加至视图
    #离散余弦变换

    # DCT weight
    def w(x, y, u, v):
        cu = 1.
        cv = 1.
        if u == 0:
            cu /= np.sqrt(2)
        if v == 0:
            cv /= np.sqrt(2)
        theta = np.pi / (2 * T)
        print("w")
        return ((2 * cu * cv / T) * np.cos((2 * x + 1) * u * theta) * np.cos((2 * y + 1) * v * theta))
    def DCTTransformer_event(self):
        print("离散余弦变换")
        img = cv2.imread(self.current_path).astype(np.float32)
        print("lisan")
        # Y = cv2.dct(img)  # 离散余弦变换
        # for i in range(0, 240):
        #     for j in range(0, 320):
        #         if i > 100 or j > 100:
        #             Y[i, j] = 0
        img = np.mean(img[..., :min(3, img.shape[-1])], axis=2)
        print('img.shape: ', img.shape)
        h, w = img.shape[:3]
        img = img[:(h // 2 * 2), :(w // 2 * 2)]
        img = img.astype(np.float)
        #     # 进行离散余弦变换
        img_dct = cv.dct(img)
        #     # 进行log处理
        img_dct_log = np.log(abs(img_dct))
        cv2.imwrite("cache/dct.jpg", img_dct_log*32)
        self.saveVariable(self.current_path, '', "cache/dct.jpg", '')
        self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewRight.setHidden(False)
        self.showPic_fun("cache/dct.jpg", 1, self.graphicsViewRight)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)

    #Radom变换
    def RadomTransformer_event(self):
        print("Radon变换")
        img= cv2.imread(self.current_path, cv2.IMREAD_UNCHANGED)
        from skimage.transform import radon
        grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        theta = np.linspace(0., 180., max(grayImage.shape), endpoint=False)
        sinogram = radon(grayImage, theta, circle=False)
        cv2.imwrite('./cache/RadomTransformer.png', sinogram)
        self.saveVariable(self.current_path, '', './cache/RadomTransformer.png', '')
        self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewRight.setHidden(False)
        self.showPic_fun("./cache/RadomTransformer.png", 1, self.graphicsViewRight)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)

    #end 变换

    # start 编辑

    #放大图片
    def zoomUp_event(self):
        print("放大")
        if (self.current_path == None):
            QMessageBox.warning(self, "提示", "您还没选择图片唉！", QMessageBox.Yes)
            return
        print("zoomscale:",self.zoomscale)
        if(self.zoomscale>=1.7):
            QMessageBox.warning(self, "提示", "图片够大了！", QMessageBox.Yes)
        else:
            # self.showPic_fun(self.current_path,self.zoomscale+0.02,self.graphicsViewCenter)
            # #self.showPic_fun(self.current_path, self.zoomscale + 0.02, self.graphicsViewLeft)
            img = cv2.imread(self.current_path.replace('\n', ''))  # 读取图像
            x = img.shape[1]  # 获取图像大小
            y = img.shape[0]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换图像通道
            self.zoomscale += 0.02  # 图片放缩尺度
            frame = QImage(img, x, y, QImage.Format_RGB888)
            pix = QPixmap.fromImage(frame)
            self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
            self.item.setScale(self.zoomscale)
            self.scene = QGraphicsScene()  # 创建场景
            self.scene.addItem(self.item)
            self.graphicsViewCenter.setScene(self.scene)  # 将场景添加至视图
            self.graphicsViewLeft.setScene(self.scene)  # 将场景添加至视图
    #缩小图片
    def zoomDown_event(self):
        print("缩小")
        if(self.current_path==None):
            QMessageBox.warning(self, "提示", "您还没选择图片唉！", QMessageBox.Yes)
            return
        if(self.zoomscale<=0.3):
            QMessageBox.warning(self, "提示", "够小了！", QMessageBox.Yes)
        else:
            # self.showPic_fun(self.current_path, self.zoomscale - 0.02, self.graphicsViewCenter)
            # #self.showPic_fun(self.current_path, self.zoomscale - 0.02, self.graphicsViewLeft)
            img = cv2.imread(self.current_path.replace('\n', ''))  # 读取图像
            x = img.shape[1]  # 获取图像大小
            y = img.shape[0]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换图像通道
            self.zoomscale -= 0.02  # 图片放缩尺度
            frame = QImage(img, x, y, QImage.Format_RGB888)
            pix = QPixmap.fromImage(frame)
            self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
            self.item.setScale(self.zoomscale)
            self.scene = QGraphicsScene()  # 创建场景
            self.scene.addItem(self.item)
            self.graphicsViewCenter.setScene(self.scene)  # 将场景添加至视图
            self.graphicsViewLeft.setScene(self.scene)  # 将场景添加至视图
    #灰度
    def grayLevel_event(self):
        print("灰度")
        if(self.current_path==None):
            QMessageBox.warning(self, "提示", "还没有选择图片！", QMessageBox.Yes)
            return
        self.graphicsViewCenter.setHidden(True)#隐藏中间显示区域
        self.closeLightSlider(True)
        self.graphicsViewRight.setHidden(False)
        self.showGrayPic_fun(self.current_path,1,self.graphicsViewRight)
        self.graphicsViewLeft.setHidden(False)
        self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)
    #亮度调节
    def lightSlider_event(self):
        img2 = cv2.imread(self.current_path.replace('\n', ''),cv2.IMREAD_UNCHANGED)  # 读取图像
        img=img2
        x = img.shape[1]  # 获取图像大小
        y = img.shape[0]
        print(self.lightSlider.value())
        # rows, cols, chunnel = img1.shape
        # blank = np.zeros([rows, cols, chunnel], img1.dtype)
        # dst = cv2.addWeighted(img1, 1, blank, -0.3, self.lightSlider.value())
        value = self.lightSlider.value() * 1.0 / 10
        rows, cols, channels = img.shape
        blank = np.zeros([rows, cols, channels], img.dtype)
        res = cv.addWeighted(img, value, blank, 1, 3)
        self.zoomscale = 1 # 图片放缩尺度
        frame = QImage(res, x, y, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
        self.item.setScale(self.zoomscale)
        self.scene = QGraphicsScene()  # 创建场景
        self.scene.addItem(self.item)
        self.graphicsViewCenter.setScene(self.scene)  # 将场景添加至视图
    def lightLevel_event(self):
        print("亮度")
        img = cv2.imread(self.current_path.replace('\n', ''), cv2.IMREAD_UNCHANGED)  # 读取图像
        #value = self.horizontalSlider.value()
        value, ok = QInputDialog.getText(self, '亮度', '请输入亮度值', QLineEdit.Normal, '10')
        value = np.int(value.replace(' ', '')) * 1.0 / 10
        x = img.shape[1]  # 获取图像大小
        y = img.shape[0]
        rows, cols, channels = img.shape
        blank = np.zeros([rows, cols, channels], img.dtype)
        res = cv.addWeighted(img, value, blank, 1, 3)
        self.zoomscale = 1  # 图片放缩尺度
        frame = QImage(res, x, y, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
        self.item.setScale(self.zoomscale)
        self.scene = QGraphicsScene()  # 创建场景
        self.scene.addItem(self.item)
        self.graphicsViewLeft.setScene(self.scene)  # 将场景添加至视图
        self.graphicsViewCenter.setScene(self.scene)  # 将场景添加至视图

    #旋转功能
    def rotate90_event(self):
        img = cv2.imread(self.current_path.replace("\n", ""),cv2.IMREAD_UNCHANGED)
        img_rotate = np.rot90(img, 1)
        self.current_idx += 1
        self.workIdx += 1
        self.showCount.setText('{}/{}'.format(self.workIdx + 1, len(self.workCache)))
        cv2.imwrite("./cache/tranformer_{}.jpg".format(self.current_idx), img_rotate)
        self.current_path = "./cache/tranformer_{}.jpg".format(self.current_idx)
        self.workCache.append(self.current_path)  # 加入工作区缓存
        # cv2.imwrite("cache/rotate90.jpg",img_rotate)
        # self.current_path="cache/rotate90.jpg"
        #self.showPicByImg_fun(img_rotate,1,self.graphicsViewCenter)
        x = img_rotate.shape[1]  # 获取图像大小
        y = img_rotate.shape[0]
        img = cv2.cvtColor(img_rotate, cv2.COLOR_BGR2RGB)  # 转换图像通道
        self.zoomscale = 1  # 图片放缩尺度
        frame = QImage(img, x, y, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
        self.item.setScale(self.zoomscale)
        self.scene = QGraphicsScene()  # 创建场景
        self.scene.addItem(self.item)
        self.graphicsViewCenter.setScene(self.scene)  # 将场景添加至视图
        self.graphicsViewLeft.setScene(self.scene)  # 将场景添加至视图

    # 逆时针旋转90度
    def rotate270_event(self):
        img = cv2.imread(self.current_path.replace("\n", ""), cv2.IMREAD_UNCHANGED)
        img_rotate = np.rot90(img, 3)
        self.current_idx += 1
        self.workIdx += 1
        self.showCount.setText('{}/{}'.format(self.workIdx + 1, len(self.workCache)))
        cv2.imwrite("./cache/tranformer_{}.jpg".format(self.current_idx), img_rotate)
        self.current_path = "./cache/tranformer_{}.jpg".format(self.current_idx)
        self.workCache.append(self.current_path)  # 加入工作区缓存f
        # cv2.imwrite("cache/rotate90.jpg", img_rotate)
        # self.current_path = "cache/rotate90.jpg"
        # self.showPicByImg_fun(img_rotate,1,self.graphicsViewCenter)
        x = img_rotate.shape[1]  # 获取图像大小
        y = img_rotate.shape[0]
        img = cv2.cvtColor(img_rotate, cv2.COLOR_BGR2RGB)  # 转换图像通道
        self.zoomscale = 1  # 图片放缩尺度
        frame = QImage(img, x, y, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
        self.item.setScale(self.zoomscale)
        self.scene = QGraphicsScene()  # 创建场景
        self.scene.addItem(self.item)
        self.graphicsViewCenter.setScene(self.scene)  # 将场景添加至视图
        self.graphicsViewLeft.setScene(self.scene)  # 将场景添加至视图

    def rotate_event(self):
        print("旋转")
        img = cv2.imread(self.current_path.replace("\n",""))

        h, w = img.shape[:2]
        center = (w // 2, h // 2)

        # 旋转中心坐标，逆时针旋转：-90°，缩放因子：1
        M_2 = cv2.getRotationMatrix2D(center, self.rotateSlider.value(), 1)
        img = cv2.warpAffine(img, M_2, (w, h))
        # cv2.imshow("./rotated_-90.jpg", rotated_2)
        x = img.shape[1]  # 获取图像大小
        y = img.shape[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换图像通道
        self.zoomscale =1  # 图片放缩尺度
        frame = QImage(img, x, y, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
        self.item.setScale(self.zoomscale)
        self.scene = QGraphicsScene()  # 创建场景
        self.scene.addItem(self.item)
        self.graphicsViewCenter.setScene(self.scene)  # 将场景添加至视图
    #截图功能
    def printScreen_event(self):
        print("截图")
        items = ["原图片自定义截图", "鼠标截图"]
        value, ok = QInputDialog.getItem(self, "截图", "请选择操作类型:", items, 0, False)
        if value == "原图片自定义截图":
            img = Image.open(self.current_path)
            img_size = img.size
            value, ok = QInputDialog.getText(self, '截图', '请输入截取宽、高的百分比', QLineEdit.Normal, '0.8 0.8')
            c, b = map(float, value.split())
            h = img_size[1]  # 图片高度
            w = img_size[0]  # 图片宽度
            x = 0.2 * w
            y = 0.2 * h
            w = c * w
            h = b * h
            out = img.crop((x, y, x + w, y + h))
            out.save('./cache/screenthand.jpg')
            self.saveVariable(self.current_path, '', './cache/screenthand.jpg', '')
            self.graphicsViewCenter.setHidden(True)  # 隐藏中间显示区域
            self.closeLightSlider(True)
            self.graphicsViewRight.setHidden(False)
            self.showPic_fun('./cache/screenthand.jpg', 1, self.graphicsViewRight)
            self.graphicsViewLeft.setHidden(False)
            self.showPic_fun(self.current_path, 1, self.graphicsViewLeft)
        elif value == "鼠标截图":
            self.screenshot = ScreenShotsWin()
            self.screenshot.showFullScreen()
            # self.saveVariable(self.current_path, '', 'cache/screenthand.jpg', '')
            # imghand = cv2.imread('cache/screenthand.jpg',cv2.IMREAD_UNCHANGED)
            # x = imghand.shape[1]  # 获取图像大小
            # y = imghand.shape[0]
            # img = cv2.cvtColor(imghand, cv2.COLOR_BGR2RGB)  # 转换图像通道
            # self.zoomscale = 1  # 图片放缩尺度
            # frame = QImage(img, x, y, QImage.Format_RGB888)
            # pix = QPixmap.fromImage(frame)
            # self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
            # self.item.setScale(self.zoomscale)
            # self.scene = QGraphicsScene()  # 创建场景
            # self.scene.addItem(self.item)
            # self.graphicsViewRight.setScene(self.scene)  # 将场景添加至视图
        else :
            return

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
        self.rightLeftButton.setHidden(not flag)
        self.leftRightButton.setHidden(not flag)
    #显示灰度图像
    def showGrayPic_fun(self, path, zoomscale, desPosition):
        #显示下一个上一个、个数按钮
        self.next.setHidden(False)
        self.back.setHidden(False)
        self.label_2.setHidden(False)
        self.label_3.setHidden(False)
        self.showCount.setHidden(False)
        self.showCount.setText('{}/{}'.format(self.workIdx + 1, len(self.workCache)))
        if path != '':
            img = cv2.imread(path.replace('\n', ''))  # 读取图像
        else:
            return
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
        # 显示下一个上一个、个数按钮
        self.next.setHidden(False)
        self.back.setHidden(False)
        self.label_2.setHidden(False)
        self.label_3.setHidden(False)
        self.showCount.setHidden(False)
        self.showCount.setText('{}/{}'.format(self.workIdx + 1, len(self.workCache)))
        if path!='':
            img = cv2.imread(path.replace('\n',''))  # 读取图像
        else:
            return
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
        # 显示下一个上一个、个数按钮
        self.next.setHidden(False)
        self.back.setHidden(False)
        self.label_2.setHidden(False)
        self.label_3.setHidden(False)
        self.showCount.setHidden(False)
        self.showCount.setText('{}/{}'.format(self.workIdx + 1, len(self.workCache)))
        if img=='':
            return
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
        self.current_path = self.history1.text().replace("\n","")
        self.workCache.append(self.current_path)
        self.workIdx +=1
        self.current_idx = len(self.source_img_list)
        self.source_img_list.append(self.current_path)
        self.graphicsViewCenter.setHidden(False)
        self.graphicsViewLeft.setHidden(True)
        self.graphicsViewRight.setHidden(True)
        # 隐藏下一个上一个、个数按钮
        self.next.setHidden(True)
        self.back.setHidden(True)
        self.label_2.setHidden(True)
        self.label_3.setHidden(True)
        self.showCount.setHidden(True)
        self.closeLightSlider(False)
        img = cv2.imread(self.current_path.replace('\n', ''))  # 读取图像
        x = img.shape[1]  # 获取图像大小
        y = img.shape[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换图像通道
        self.zoomscale = 1  # 图片放缩尺度
        frame = QImage(img, x, y, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
        self.item.setScale(self.zoomscale)
        self.scene = QGraphicsScene()  # 创建场景
        self.scene.addItem(self.item)
        self.graphicsViewCenter.setScene(self.scene)  # 将场景添加至视图
    def setHistory2_event(self):
        self.current_path = self.history2.text().replace("\n","")
        self.workCache.append(self.current_path)
        self.workIdx += 1
        self.current_idx = len(self.source_img_list)
        self.source_img_list.append(self.current_path)
        # 隐藏下一个上一个、个数按钮
        self.next.setHidden(True)
        self.back.setHidden(True)
        self.label_2.setHidden(True)
        self.label_3.setHidden(True)
        self.showCount.setHidden(True)
        self.graphicsViewCenter.setHidden(False)
        self.graphicsViewLeft.setHidden(True)
        self.graphicsViewRight.setHidden(True)
        self.closeLightSlider(False)
        img = cv2.imread(self.current_path.replace('\n', ''))  # 读取图像
        x = img.shape[1]  # 获取图像大小
        y = img.shape[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换图像通道
        self.zoomscale = 1  # 图片放缩尺度
        frame = QImage(img, x, y, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
        self.item.setScale(self.zoomscale)
        self.scene = QGraphicsScene()  # 创建场景
        self.scene.addItem(self.item)
        self.graphicsViewCenter.setScene(self.scene)  # 将场景添加至视图
    def setHistory3_event(self):
        self.current_path = self.history3.text().replace("\n","")
        self.workCache.append(self.current_path)
        self.workIdx += 1
        self.current_idx = len(self.source_img_list)
        self.source_img_list.append(self.current_path)
        # 隐藏下一个上一个、个数按钮
        self.next.setHidden(True)
        self.back.setHidden(True)
        self.label_2.setHidden(True)
        self.label_3.setHidden(True)
        self.showCount.setHidden(True)
        self.graphicsViewCenter.setHidden(False)
        self.graphicsViewLeft.setHidden(True)
        self.graphicsViewRight.setHidden(True)
        self.closeLightSlider(False)
        img = cv2.imread(self.current_path.replace('\n', ''))  # 读取图像
        x = img.shape[1]  # 获取图像大小
        y = img.shape[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换图像通道
        self.zoomscale = 1 # 图片放缩尺度
        frame = QImage(img, x, y, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
        self.item.setScale(self.zoomscale)
        self.scene = QGraphicsScene()  # 创建场景
        self.scene.addItem(self.item)
        self.graphicsViewCenter.setScene(self.scene)  # 将场景添加至视图

    #打开图片
    def open_event(self):
        openfile_name = QFileDialog.getOpenFileName(self, '选择文件', 'D:/Picture',
                                                        'Image files(*.jpg , *.png)')
        if openfile_name[0]:
            self.imgCache_add(openfile_name[0])#将读取到的图片路径加入到缓存中
            self.current_path = openfile_name[0]
            self.workCache.append(self.current_path)
            self.workIdx += 1
            self.current_idx = len(self.source_img_list)
            self.source_img_list.append(self.current_path)
            self.next.setHidden(True)
            self.back.setHidden(True)
            self.label_2.setHidden(True)
            self.label_3.setHidden(True)
            self.showCount.setHidden(True)
            self.graphicsViewCenter.setHidden(False)
            self.graphicsViewLeft.setHidden(True)
            self.graphicsViewRight.setHidden(True)
            self.closeLightSlider(False)
            img = cv2.imread(self.current_path.replace('\n', ''))  # 读取图像
            x = img.shape[1]  # 获取图像大小
            y = img.shape[0]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换图像通道
            self.zoomscale = 1  # 图片放缩尺度
            frame = QImage(img, x, y, QImage.Format_RGB888)
            pix = QPixmap.fromImage(frame)
            self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
            self.item.setScale(self.zoomscale)
            self.scene = QGraphicsScene()  # 创建场景
            self.scene.addItem(self.item)
            self.graphicsViewCenter.setScene(self.scene)  # 将场景添加至视图
        else:
            QMessageBox.warning(self,"提示","这个图片好像逃跑了！",QMessageBox.Yes)
    #保存图片
    def save_event(self):
        if  not self.current_path:
            QMessageBox.warning(self,"提示","什么都没有，我保存什么？",QMessageBox.Yes)
            return
        file_path = QFileDialog.getSaveFileName(self, '选择保存位置', 'D:/Picture/*.png',
                                                'Image files(*.png)')
        file_path = file_path[0]
        if file_path:
            cv.imwrite(file_path, cv.imread(self.current_path))

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
