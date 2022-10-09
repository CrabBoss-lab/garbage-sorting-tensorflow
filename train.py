# Author:yujunyu
# -*- codeing = utf-8 -*-
# @Time :2022/8/18 23:18
# @Author :yujunyu
# @Site :
# @File :train.py
# @software: PyCharm

"""
总共6类。纸板：0，玻璃：glass, 金属：metal，纸：paper，塑料：plastic, 一般垃圾：trash
"""

import numpy as np
import matplotlib.pyplot as plt
import glob,os,random
import tensorflow as tf
from PIL import Image

# 导入数据
base_path = "./dataset"
# 随机查看一张图
img_list = glob.glob(os.path.join(base_path, "*/*.jpg"))    # 获取指定路径下的所有
# 图片
# print(len(img_list))  # 共2295张数据
# i=random.randint(0,2294)
# print(img_list[i])
# plt.imshow(Image.open(img_list[i]))
# plt.show()

# 对数据进行划分及增强（从文件夹创建和加载数据集）
train_data = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255.,    # RGB（0-255——》0-1）
    shear_range=0.1,    # 随机应用剪切变换
    zoom_range=0.2,     # 随机缩
    rotation_range=40,
    width_shift_range=0.2,  # 在垂直或水平方向上随机平移图片的范围
    height_shift_range=0.2,
    horizontal_flip=True,   # 进行随机水平翻转
    vertical_flip=True,     # 进行随机竖直翻转
    validation_split=0.25   # 分割数据作为验证数据
)

batch_size=32
train_generator = train_data.flow_from_directory(
    base_path,  # 目录-每个类应该包含一个子目录
    target_size=(300, 300),  # 目标大小-调整目标图像大小
    batch_size=batch_size,     # 批次大小-每个batch的数据批次
    class_mode='categorical',   # 类模式-标签的类型。分类（categorical）、二进制（“binary”）等等。默认是分类
    subset='training'   # 子集-如果在ImageDataGenerator中设置了验证分割（validation_split），则为数据的子集（训练集"training" 或验证集"validation"）
)
validation_generator = train_data.flow_from_directory(
    base_path,
    target_size=(300,300),
    batch_size=batch_size,
    shuffle=False,
    class_mode="categorical",
    seed=0,
    subset="validation",
)


labels = (train_generator.class_indices)       # class_indices可以获取包含类名到类索引的字典例如，cardboard对应0，glass对应1。。。
labels = dict((v,k) for k,v in labels.items())
print(labels)   # # {0: '0', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}

# 模型搭建
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu',input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D(pool_size=2),

    tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),

    tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),

    tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(units=64, activation='relu'),

    tf.keras.layers.Dense(units=6, activation='softmax')
])

model.summary()


lr=0.001
epoch=100
model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(learning_rate=lr), metrics=['accuracy'])
# History=model.fit_generator(train_generator, epochs=epoch, steps_per_epoch=2068//batch_size,validation_data=validation_generator,validation_steps=227//batch_size)
History=model.fit(train_generator,epochs=epoch,validation_data=validation_generator)
model.save('model_5.h5')


# 保存训练时的loss、accuracy的变化图
plt.plot(np.arange(0, epoch), History.history["loss"], label="train_loss")
plt.plot(np.arange(0, epoch), History.history["accuracy"],label='train_accuracy')
plt.plot(np.arange(0, epoch), History.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epoch), History.history["val_accuracy"],label='val_accuracy')
plt.xlabel("Epoch #")
plt.ylabel("Loss/accuracy")
plt.legend(loc='best')
plt.savefig("loss_acc_5.png")
plt.show()