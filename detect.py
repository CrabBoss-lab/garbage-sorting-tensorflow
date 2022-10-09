# Author:yujunyu
# -*- codeing = utf-8 -*-
# @Time :2022/8/17 12:36
# @Author :yujunyu
# @Site :
# @File :detect.py
# @software: PyCharm
"""
传入垃圾图片进行预测
"""
import tensorflow as tf
from PIL import Image
import numpy as np

img_filepath = 'detect_img/img_3.png'
model = tf.keras.models.load_model('./model/model_epo100.h5')
img = Image.open(img_filepath)
img = np.array(img.resize((300, 300)))
img = img / 255.
image = img.reshape((1, 300, 300, 3))
probabilities = model.predict(image)    ## 预测输出每个类别的概率
# print(probabilities)
prediction = np.argmax(probabilities)  # 找出最大值
prediction_1=np.max(probabilities)
# print(prediction)
label_list = ['纸板-cradboard', '玻璃-glass', '金属-metal', '纸-paper', '塑料-plastic', '其他垃圾-trash']
for i in range(len(label_list)):
    res = (f"类别{i+1}:"+label_list[i], "概率:{:.2%}".format(probabilities[0][i]))
    print(" ".join('%s' %a for a in res))
    # print('——类别:{} 概率:{:.2%}——'.format(label_list[i],probabilities[0][i]))
for j in range(len(label_list)):
    if int(prediction)==j:
        print("预测结果: 类别：{}   概率:{:.2%}".format(label_list[prediction], prediction_1))
    else:
        pass



