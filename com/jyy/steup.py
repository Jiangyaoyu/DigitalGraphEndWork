from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import *
import sys

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

        #初始化隐藏组件
        self.next.setHidden(True)
        self.back.setHidden(True)
        self.graphicsViewLeft.setHidden(True)
        self.graphicsViewRight.setHidden(True)
        self.graphicsViewCenter.setHidden(True)
        self.graphicsViewCenter.setStyleSheet("border: 0px;background-color:#F0F0F0")

        # 方法绑定
        self.fileopen.triggered.connect(self.open_event)
        self.recentBrowse.triggered.connect(self.recentBrowse.trigger)
        self.exit.triggered.connect(self.exit.trigger)
        self.filesave.triggered.connect(self.save_event)
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

        # 设置成员变量
        self.current_img = None
        self.current_idx = -1
        self.target_img = None
        self.source_img_list = []

    #显示图片
    def showPic_fun(self,path,zoomscale,desPosition):
        img = cv.imread(path)  # 读取图像
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # 转换图像通道
        x = img.shape[1]  # 获取图像大小
        y = img.shape[0]
        self.zoomscale = zoomscale # 图片放缩尺度
        frame = QImage(img, x, y, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
        self.item.setScale(self.zoomscale)
        #desPosition.setGeometry(desPosition.pos().x(), desPosition.pos().x(), x*zoomscale, y*zoomscale)
        self.scene = QGraphicsScene()  # 创建场景
        self.scene.addItem(self.item)
        desPosition.setScene(self.scene)  # 将场景添加至视图

    #打开图片
    def open_event(self):
        openfile_name = QFileDialog.getOpenFileName(self, '选择文件', 'D:/Picture',
                                                        'Image files(*.jpg , *.png)')
        if openfile_name[0]:
            self.current_img = openfile_name[0]
            self.current_idx = len(self.source_img_list)
            self.source_img_list.append(self.current_img)
            self.graphicsViewCenter.setHidden(False)
            self.showPic_fun(self.current_img,0.5,self.graphicsViewCenter)
        else:
            print('No file opened.')
    def filesave_event(self):
        print("save")

    #保存图片
    def save_event(self):
        if  not self.current_img:
            QMessageBox.warning(self,"提示","选图片了吗？",QMessageBox.Yes)
            return
        file_path = QFileDialog.getSaveFileName(self, '选择保存位置', 'D:/Picture/*.png',
                                                'Image files(*.png)')
        print('file_path: ', file_path)
        file_path = file_path[0]
        if file_path:
            print('file_path: ', file_path)
            cv.imwrite(file_path, self.current_img)
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    mainWindon = MainWindow()
    mainWindon.show()

    sys.exit(app.exec_())
