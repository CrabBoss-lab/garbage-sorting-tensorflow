# Author:yujunyu
# -*- codeing = utf-8 -*-
# @Time :2022/8/18 17:53
# @Author :yujunyu
# @Site :
# @File :predict.py
# @software: PyCharm

"""
使用验证集进行预测
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import glob,os,random
from PIL import Image

model=tf.keras.models.load_model("./model/model_epo100.h5")

base_path = "./dataset"
img_list = glob.glob(os.path.join(base_path, "*/*.jpg"))

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, shear_range=0.1, zoom_range=0.1,
    width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True,
    vertical_flip=True, validation_split=0.1)
test_data = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,validation_split=0.1)

train_generator = train_datagen.flow_from_directory(
    base_path, target_size=(300, 300), batch_size=16,
    class_mode='categorical', subset='training', seed=0)

validation_generator = test_data.flow_from_directory(base_path,target_size=(300,300),
                  batch_size=16,class_mode="categorical",subset="validation",seed=0 )

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())


# 预测
test_x, test_y = validation_generator.__getitem__(random.randint(1,16))
preds = model.predict(test_x)
plt.figure(figsize=(16, 16))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.title('pred:%s / truth:%s' % (labels[np.argmax(preds[i])], labels[np.argmax(test_y[i])]))
    plt.imshow(test_x[i])
plt.show()
