#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
from scipy.io import loadmat
from sklearn import preprocessing  # 0-1编码
from sklearn.model_selection import StratifiedShuffleSplit  # 随机划分，保证每一类比例相同

class preprocess(object):
    
    def __init__(self):
        self.d_path='/Users/tung/Python/WorkProject/PHMresearch/WDCNN&LR_FaultDiagnosis/data/0HP'
        self.length=2048
        self.number=200
        self.normal=True
        self.rate=[0.5, 0.25, 0.25]
        self.enc = True
        self.enc_step = 28
        self.filenames = []

    def capture(self):
        """读取mat文件，返回字典

        :param original_path: 读取路径
        :return: 数据字典
        """
        files = {}
        for i in self.filenames:
            # 文件路径
            file_path = os.path.join(self.d_path, i)
            file = loadmat(file_path)
            file_keys = file.keys()
            for key in file_keys:
                if 'DE' in key:
                    files[i] = file[key].ravel()
        return files

    def slice_enc(self, data):
        """将数据切分为前面多少比例，后面多少比例.

        :param data: 单挑数据
        :param slice_rate: 验证集以及测试集所占的比例
        :return: 切分好的数据
        """
        slice_rate = self.rate[1] + self.rate[2]
        keys = data.keys()
        Train_Samples = {}
        Test_Samples = {}
        for i in keys:
            slice_data = data[i]
            all_lenght = len(slice_data)
            end_index = int(all_lenght * (1 - slice_rate))
            samp_train = int(self.number * (1 - slice_rate))  # 700
            Train_sample = []
            Test_Sample = []
            if self.enc:
                enc_time = self.length // self.enc_step
                samp_step = 0  # 用来计数Train采样次数
                for j in range(samp_train):
                    random_start = np.random.randint(low=0, high=(end_index - 2 * self.length))
                    label = 0
                    for h in range(enc_time):
                        samp_step += 1
                        random_start += self.enc_step
                        sample = slice_data[random_start: random_start + self.length]
                        Train_sample.append(sample)
                        if samp_step == samp_train:
                            label = 1
                            break
                    if label:
                        break
            else:
                for j in range(samp_train):
                    random_start = np.random.randint(low=0, high=(end_index - self.length))
                    sample = slice_data[random_start:random_start + self.length]
                    Train_sample.append(sample)

            # 抓取测试数据
            for h in range(self.number - samp_train):
                random_start = np.random.randint(low=end_index, high=(all_lenght - self.length))
                sample = slice_data[random_start:random_start + self.length]
                Test_Sample.append(sample)
            Train_Samples[i] = Train_sample
            Test_Samples[i] = Test_Sample
        return Train_Samples, Test_Samples

    # 仅抽样完成，打标签
    def add_labels(self, train_test):
        X = []
        Y = []
        label = 0
        for i in self.filenames:
            x = train_test[i]
            X += x
            lenx = len(x)
            Y += [label] * lenx
            label += 1
        return X, Y

    # one-hot编码
    def one_hot(self, Train_Y, Test_Y):
        Train_Y = np.array(Train_Y).reshape([-1, 1])
        Test_Y = np.array(Test_Y).reshape([-1, 1])
        Encoder = preprocessing.OneHotEncoder()
        Encoder.fit(Train_Y)
        Train_Y = Encoder.transform(Train_Y).toarray()
        Test_Y = Encoder.transform(Test_Y).toarray()
        Train_Y = np.asarray(Train_Y, dtype=np.int32)
        Test_Y = np.asarray(Test_Y, dtype=np.int32)
        return Train_Y, Test_Y

    def scalar_stand(self, Train_X, Test_X):
        # 用训练集标准差标准化训练集以及测试集
        scalar = preprocessing.StandardScaler().fit(Train_X)
        Train_X = scalar.transform(Train_X)
        Test_X = scalar.transform(Test_X)
        return Train_X, Test_X

    def valid_test_slice(self, Test_X, Test_Y):
        test_size = self.rate[2] / (self.rate[1] + self.rate[2])
        ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        for train_index, test_index in ss.split(Test_X, Test_Y):
            X_valid, X_test = Test_X[train_index], Test_X[test_index]
            Y_valid, Y_test = Test_Y[train_index], Test_Y[test_index]
            return X_valid, Y_valid, X_test, Y_test

    def prepro(self, d_path, length, number, normal, rate, enc, enc_step):
        self.d_path=d_path
        self.length=length
        self.number=number
        self.normal=normal
        self.rate=rate
        self.enc=enc
        self.enc_step=enc_step
        
        # 获得该文件夹下所有.mat文件名
        self.filenames = os.listdir(self.d_path)
        del self.filenames[2]   #删除.DS_Store
        
        # 从所有.mat文件中读取出数据的字典
        data = self.capture()
        # 将数据切分为训练集、测试集
        train, test = self.slice_enc(data)
        # 为训练集制作标签，返回X，Y
        Train_X, Train_Y = self.add_labels(train)
        # 为测试集制作标签，返回X，Y
        Test_X, Test_Y = self.add_labels(test)
        # 为训练集Y/测试集One-hot标签
        Train_Y, Test_Y = self.one_hot(Train_Y, Test_Y)
        # 训练数据/测试数据 是否标准化.
        if normal:
            Train_X, Test_X = self.scalar_stand(Train_X, Test_X)
        else:
            # 需要做一个数据转换，转换成np格式.
            Train_X = np.asarray(Train_X)
            Test_X = np.asarray(Test_X)
        # 将测试集切分为验证集合和测试集.
        Valid_X, Valid_Y, Test_X, Test_Y = self.valid_test_slice(Test_X, Test_Y)
        return Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y


if __name__ == "__main__":
  
    """对数据进行预处理,返回train_X, train_Y, valid_X, valid_Y, test_X, test_Y样本.
        
    :param d_path: 源数据地址
    :param length: 信号长度，默认5个信号周期T=400*10S=66.6min～1h，2048
    :param number: 每种信号个数,总共9类,默认每个类别1000个数据
    :param normal: 是否标准化.True,Fales.默认True
    :param rate: 训练集/验证集/测试集比例.默认[0.5,0.25,0.25],相加要等于1
    :param enc: 训练集、验证集是否采用数据增强.Bool,默认True
    :param enc_step: 增强数据集采样顺延间隔
    :return: Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y
    
    """
    #传感器研发组 做实验、售后去现场维护、采回来的数据
    #溶液地未连接(100)、流通池接地(100)、电缆线未连接(100)、球泡破裂(35)、支架损坏(31)、
    #电极污染(36）、电解液缺失(50)、水样波动(100)、正常
    #重采样之后 200*9 = 1800
    
    path = '/Users/tung/Python/WorkProject/PHMresearch/WDCNN&LR_FaultDiagnosis/data/0HP'
    test = preprocess()
    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = test.prepro(d_path=path,
                                                                length=2048,
                                                                number=200,
                                                                normal=False,
                                                                rate=[0.7, 0.2, 0.1],
                                                                enc=False,
                                                                enc_step=28)
    print('训练样本维度:', train_X.shape)
    print(train_X.shape[0], '训练样本个数')
    print('训练标签one-hot维度',train_Y.shape[1])
    print('验证样本的维度', valid_X.shape)
    print(valid_X.shape[0], '验证样本个数')
    print('测试样本的维度', test_X.shape)
    print(test_X.shape[0], '测试样本个数')
