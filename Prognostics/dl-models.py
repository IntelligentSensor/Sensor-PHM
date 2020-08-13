#!/usr/bin/env python

import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from preprocess import preprocess

import sys
import keras as K
import tensorflow as tf
from datetime import datetime
from keras.regularizers import l2
from keras.utils import plot_model
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import TensorBoard
from keras.models import model_from_json
from keras.layers import Input, Dense, Flatten, Activation
from keras.layers import Conv1D, Dropout, BatchNormalization, MaxPooling1D

from keras.layers import LSTM, Bidirectional
from keras.layers.core import Flatten, Dense, Dropout

py_ver = sys.version
k_ver = K.__version__
tf_ver = tf.__version__

K.backend.clear_session()

print("Using Python version " + str(py_ver))
print("Using Keras version " + str(k_ver))
print("Using TensorFlow version " + str(tf_ver))

Class_dict={0:'正常', 1:'溶液地未连接', 2:'流通池接地', 3:'电缆线未连接', 4:'球泡破裂', 5:'支架损坏',
    6:'电极污染', 7:'电解液缺失', 8:'水样波动'}

# 训练参数
batch_size = 10
epochs = 30
num_classes = 9
length = 2048
BatchNorm = True        # 是否批量归一化
number = 200            # 每类样本的数量
normal = True           # 是否标准化
rate = [0.7,0.2,0.1]    # 测试集验证集划分比例

path = '/Users/tung/Python/WorkProject/PHMresearch/WDCNN&LR_FaultDiagnosis/'
preprocess = preprocess()

x_train, y_train, x_valid, y_valid, x_test, y_test = preprocess.prepro(d_path=path+'data/0HP',length=length,
                                                                       number=number,
                                                                       normal=normal,
                                                                       rate=rate,
                                                                       enc=True, enc_step=340)

# 输入卷积的时候还需要修改一下，增加通道数目
x_train, x_valid, x_test = x_train[:,:,np.newaxis], x_valid[:,:,np.newaxis], x_test[:,:,np.newaxis]

# 输入数据的维度
input_shape =x_train.shape[1:]

print('训练样本维度:', x_train.shape)
print(x_train.shape[0], '训练样本个数')
print('验证样本的维度', x_valid.shape)
print(x_valid.shape[0], '验证样本个数')
print('测试样本的维度', x_test.shape)
print(x_test.shape[0], '测试样本个数')

def get_label(row):
    for c in range(len(row)):
        if row[c]==1:
            return c

def decode(arr):
    temp = np.zeros(len(arr))
    for i in range(len(arr)):
        temp[i] = get_label(arr[i])
    return temp
y_test_decode = decode(y_test)
y_train_decode = decode(y_train)

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

'WDCNN'
# 自定义卷积层wdcnn
def wdcnn(filters, kernerl_size, strides, conv_padding, pool_padding,  pool_size, BatchNormal):
    """wdcnn层神经元
        
        :param filters: 卷积核的数目，整数
        :param kernerl_size: 卷积核的尺寸，整数
        :param strides: 步长，整数
        :param conv_padding: 'same','valid'
        :param pool_padding: 'same','valid'
        :param pool_size: 池化层核尺寸，整数
        :param BatchNormal: 是否Batchnormal，布尔值
        :return: model
        """
    model.add(Conv1D(filters=filters, kernel_size=kernerl_size, strides=strides,
                     padding=conv_padding, kernel_regularizer=l2(1e-4)))
    if BatchNormal:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=pool_size, padding=pool_padding))
    return model

# 实例化序贯模型
model = Sequential()
# 搭建输入层，第一层卷积。因为要指定input_shape，所以单独放出来
model.add(Conv1D(filters=16, kernel_size=64, strides=16, padding='same',kernel_regularizer=l2(1e-4), input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

#0-1之间均匀分布的Dropout
# model.add(Dropout( np.round(random.uniform(0,1), 2) ))

# 第二层卷积
model = wdcnn(filters=32, kernerl_size=3, strides=1, conv_padding='same',
              pool_padding='valid',  pool_size=2, BatchNormal=BatchNorm)
# 第三层卷积
model = wdcnn(filters=64, kernerl_size=3, strides=1, conv_padding='same',
              pool_padding='valid', pool_size=2, BatchNormal=BatchNorm)
# 第四层卷积
model = wdcnn(filters=64, kernerl_size=3, strides=1, conv_padding='same',
              pool_padding='valid', pool_size=2, BatchNormal=BatchNorm)
# 第五层卷积
model = wdcnn(filters=64, kernerl_size=3, strides=1, conv_padding='valid',
              pool_padding='valid', pool_size=2, BatchNormal=BatchNorm)
# 从卷积到全连接需要展平
model.add(Flatten())

# 添加全连接层
model.add(Dense(units=90, activation='relu', kernel_regularizer=l2(1e-4)))
# 增加输出层
model.add(Dense(units=num_classes, activation='softmax', kernel_regularizer=l2(1e-4)))
model.summary()

# 编译模型 评价函数和损失函数相似，不过评价函数的结果不会用于训练过程中
model.compile(optimizer='Adam', loss='categorical_crossentropy',
              metrics=['accuracy'])  #metrics=[auc]

start = datetime.now()

# TensorBoard调用查看一下训练情况
tb_cb = TensorBoard(log_dir='logs')

# 开始模型训练
history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(x_valid, y_valid), shuffle=True,
                    callbacks=[tb_cb])

print("This took ", datetime.now() - start)

#变dropout率
#BN与训练速度和识别率
#样本量与识别率及标准差的关系
#对输入数据添加高斯白噪声
def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)

test = x_train[0]
testwgn = wgn(test, 10).reshape((2048, 1))  #-4dB～10dB
temp = test + testwgn

#第一层卷积核大小与抗噪
#feature map特征可分性
#保存模型
model_path = path + 'models/wdcnn.h5'
model.save(model_path)
del model

# 模型包含一个自定义 wdcnn 类的实例
model = load_model(path+'models/wdcnn.h5', custom_objects={'wdcnn': wdcnn})
model.summary()
#fine-tune

#evaluation
score = model.evaluate(x=x_test, y=y_test, verbose=0)
print("测试集上的损失：", score[0])
print("测试集上的损失:",score[1])
plot_model(model=model, to_file=path+'models/wdcnn.png', show_shapes=True)

#prediction
start = datetime.now()

unknown = x_test[0].reshape((1, 2048, 1))
predicted = model.predict(unknown)
print("Using model to predict fault for features: ")
print(unknown)
print("\nPredicted softmax vector is: ")
print(predicted)
print("\nPredicted fault is: ")
print(Class_dict[np.argmax(predicted)])

print("This took ", datetime.now() - start)

'LSTM'
x_train = x_train.reshape((x_train.shape[0], 16, 128))        #time_step、input_dim
x_valid = x_valid.reshape((x_valid.shape[0], 16, 128))
x_test = x_test.reshape((x_test.shape[0], 16, 128))

model = Sequential()

#隐藏层设置为10, input_shape（time_step、input_dim） stateful=True使用状态RNN
model.add(LSTM(units=9, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(BatchNormalization())

#全连接层，输出单个类，units为num_classes
model.add(Dense(units=num_classes, activation='softmax', kernel_regularizer=l2(1e-4)))

#告诉模型输入的格式
model.build((None, x_train.shape[1], x_train.shape[2]))      #time_step、input_dim

# #重置网络中所有层的状态
# model.reset_states()

# #重置指定层的状态
# model.layers[0].reset_states()
model.summary()

#损失函数为交叉熵，优化器为Adam，学习率为0.001
model.compile(loss='categorical_crossentropy',optimizer='Adam', metrics=['acc'])

start = datetime.now()
history =model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_valid, y_valid))#训练模型并进行测试

print("This took ", datetime.now() - start)

#保存模型
model_path = path+'models/LSTM.h5'
model.save(model_path)
del model

model = load_model(path+'models/LSTM.h5')
model.summary()

#evaluation
score = history.model.evaluate(x=x_test, y=y_test, verbose=0)
print("测试集上的损失：", score[0])
print("测试集上的损失:",score[1])
plot_model(model=model, to_file=path+'models/LSTM.png', show_shapes=True)

#prediction
start = datetime.now()

unknown = x_test[0].reshape((1, 16, 128))
predicted = model.predict(unknown)
print("Using model to predict species for features: ")
print(unknown)
print("\nPredicted softmax vector is: ")
print(predicted)
print("\nPredicted fault is: ")
print(Class_dict[np.argmax(predicted)])

print("This took ", datetime.now() - start)

'biLSTM'
model = Sequential()
#隐藏层设置为10, input_shape元组第二个参数指
model.add(Bidirectional(LSTM(units=9, input_shape=(x_train.shape[1], x_train.shape[2]))))   # activation='tanh'
model.add(BatchNormalization())

#全连接层，输出单个类，units为num_classes
model.add(Dense(units=num_classes, activation='softmax', kernel_regularizer=l2(1e-4)))

#告诉模型输入的格式
model.build((None, x_train.shape[1], x_train.shape[2]))      #time_step、input_dim
model.summary()

#损失函数为交叉熵，优化器为Adam，学习率为0.001
model.compile(loss='categorical_crossentropy',optimizer='Adam', metrics=['acc'])

start = datetime.now()
history =model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_valid, y_valid))#训练模型并进行测试

print("This took ", datetime.now() - start)

#嵌套网络保存
model.save_weights(path+'models/biLSTM.h5')
model.load_weights(path+'models/biLSTM.h5',by_name=True)
json_string = model.to_json()
model=model_from_json(json_string)
model.build((None, x_train.shape[1], x_train.shape[2]))      #time_step、input_dim
model.summary()

#evaluation
score = history.model.evaluate(x=x_test, y=y_test, verbose=0)
print("测试集上的损失：", score[0])
print("测试集上的损失:",score[1])
plot_model(model=model, to_file=path+'models/biLSTM.png', show_shapes=True)

#prediction
start = datetime.now()

unknown = x_test[0].reshape((1, 16, 128))
predicted = model.predict(unknown)
print("Using model to predict species for features: ")
print(unknown)
print("\nPredicted softmax vector is: ")
print(predicted)
print("\nPredicted fault is: ")
print(Class_dict[np.argmax(predicted)])

print("This took ", datetime.now() - start)
