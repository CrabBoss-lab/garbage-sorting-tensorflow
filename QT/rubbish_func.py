# Author:yujunyu
# -*- codeing = utf-8 -*-
# @Time :2022/8/19 16:01
# @Author :yujunyu
# @Site :
# @File :rubbish_func.py
# @software: PyCharm

from PyQt5.QtWidgets import QDialog, QFileDialog, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.Qt import Qt
import cv2
import sys, os
import tensorflow as tf
from PIL import Image
import numpy as np

from QT.readfile import ReadFile       # 导入readfile文件
from QT.rubbish_ui import Ui_Dialog    # 连接rubbish_ui文件（GUI界面）

# 窗口类
class RubbishDlg(QDialog):
    # 构造方法
    def __init__(self):
        super(RubbishDlg, self).__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        self.read = ReadFile()

        self.filepath = ""

        self.modelpath = "../model/model_epo100.h5"

    def closeEvent(self, e):
        # 窗体关闭前的释放工作,条件不满足可以阻止窗体关闭
        sys.exit(0)

    def keyPressEvent(self, e):
        # 阻止按照ESC键关闭窗体
        pass

    # 显示图像
    def images_show(self, h, w, c, data):
        image = QImage(data, w, h, w * c, QImage.Format_RGB888)
        pix = QPixmap.fromImage(image)
        # 获取QLabel的大小
        width = self.ui.label_img.width()
        height = self.ui.label_img.height()
        # 等比例放缩图片
        scaledPixmap = pix.scaled(width, height, Qt.KeepAspectRatio)
        self.ui.label_img.setPixmap(scaledPixmap)

    # 槽函数，上传功能
    def load_image(self):
        self.ui.textEdit_3.setText("")  # 再次上传一张图，清空识别

        image_format = ['jpg', 'png', 'bmp']
        self.filepath, imgtype = QFileDialog.getOpenFileName(None, "加载图片", "./", "(*.jpg  *.png *.bmp)")
        if not len(self.filepath):
            QMessageBox.information(self, "提示", "请加载需要识别的图片！", QMessageBox.Yes | QMessageBox.No)
        else:
            for i in range(len(image_format)):
                ret = self.filepath.find(image_format[i])
                break
            if not ret:
                QMessageBox.information(self, "提示", "请加载正确的图片！", QMessageBox.Yes | QMessageBox.No)
            else:
                self.ui.textEdit_2.setText("文件位置:" + self.filepath)

                data = self.read.readFile(self.filepath)
                img_data = data[3]
                img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
                h, w, c = img_data.shape
                self.images_show(h, w, c, img_data.tobytes())

    # 槽函数：识别功能
    def shibie(self):
        img_path = (self.filepath)
        print(f"图片位置:{img_path}")
        img_filepath =img_path
        model = tf.keras.models.load_model(self.modelpath)
        img = Image.open(img_filepath)
        img = np.array(img.resize((300, 300)))
        img = img / 255.
        image = img.reshape((1, 300, 300, 3))
        probabilities = model.predict(image)  ## 预测输出每个类别的概率
        # print(probabilities)
        prediction = np.argmax(probabilities)  # 找出最大值
        prediction_1 = np.max(probabilities)
        # print(prediction)
        label_list = ['纸板-0', '玻璃-glass', '金属-metal', '纸-paper', '塑料-plastic', '其他垃圾-trash']
        for i in range(len(label_list)):
            res = ("类别{}:{}".format(i+1,label_list[i]), "概率:{:.2%}".format(probabilities[0][i]))
            # print(res)
            self.ui.textEdit_3.append((" ".join('%s' %a for a in res)))
            self.ui.textEdit_3.append("\n")
            # print('——类别:{} 概率:{:.2%}——'.format(label_list[i],probabilities[0][i]))
        for j in range(len(label_list)):
            if int(prediction) == j:
                res=("预测结果: 类别{}:{}".format(j+1,label_list[prediction]),"概率:{:.2%}".format(prediction_1))
                self.ui.textEdit_3.append((" ".join('%s' %a for a in res)))
            else:
                pass
