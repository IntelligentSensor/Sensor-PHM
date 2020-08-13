#!/usr/bin/env python

import numpy as np
import pandas as pd
import scipy.stats as sts
from sklearn.utils import class_weight

import pywt
import math
import scipy.ndimage
from math import floor, log
from sklearn import preprocessing

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from preprocess import preprocess
import matplotlib.pyplot as plt

class featureEngineer(object):
    def __init__(self):
        self.root = '/Users/tung/Python/WorkProject/PHMresearch/WDCNN&LR_FaultDiagnosis/'
        #业务规则特征
        self.form_1 = np.loadtxt(self.root + "data/form/form_1.txt")        #溶液地未连接
        self.form_2 = np.loadtxt(self.root + "data/form/form_2.txt")        #流通池接地
        self.form_3 = np.loadtxt(self.root + "data/form/form_3.txt")        #电缆线未连接
        self.form_4 = np.loadtxt(self.root + "data/form/form_4.txt")        #球泡破裂
        self.form_5 = np.loadtxt(self.root + "data/form/form_5.txt")        #支架损坏
        self.form_6 = np.loadtxt(self.root + "data/form/form_6.txt")        #电极污染
        self.form_7 = np.loadtxt(self.root + "data/form/form_7.txt")        #电解液缺失
        self.form_8 = np.loadtxt(self.root + "data/form/form_8.txt")        #水样波动
        self.normal = np.loadtxt(self.root + "data/form/normal.txt")        #正常
    
    'one-hot decode'
    def decode(self, arr):
        def get_label(row):
            for c in range(len(row)):
                if row[c]==1:
                    return c
    
        temp = np.zeros(len(arr))
        for i in range(len(arr)):
            temp[i] = get_label(arr[i])
        return temp
    
    '处理样本不均衡'
    ######################overSampling####################
    def overSampling(self, data, label):
        #RandomOverSampler 随机过采样
        def Ros(data,label):   #通过简单的随机采样少数类的样本, 使得每类样本的比例为1:1:1:1
            from imblearn.over_sampling import RandomOverSampler
            
            ros = RandomOverSampler(random_state = 0)
            data_ros, label_ros = ros.fit_sample(data, label)
            return data_ros, label_ros

        #SMOTE 默认生成1:1的数据
        def Smote(data,label):  #对于少数类样本a, 随机选择一个最近邻的样本b, 然后从a与b的连线上随机选取一个点c作为新的少数类样本
            from imblearn.over_sampling import SMOTE
            
            smote = SMOTE(random_state = 0)
            data_smote, label_smote =smote.fit_sample(data, label)
            return data_smote, label_smote

        #SMOTE的变体：Borderline
        def Smote_bd(data,label):  #样本的近邻至少有一半是其他类，（此时样本被称为危险样本）最近邻中的随机样本b与该少数类样本a来自于不同的类
            from imblearn.over_sampling import BorderlineSMOTE
            
            smote = BorderlineSMOTE(random_state = 0)
            data_smote_bd, label_smote_bd =smote.fit_sample(data, label)
            return data_smote_bd, label_smote_bd

    #######################calssWeight####################
    def calssWeight(self, y_train):
        weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
        return weights

    '样本分布'
    def sampleDistribution(self, sample):     #非参数检验
        print('describe', sample.describe())  #基本统计
        df = pd.DataFrame(sample)             #离群点
        df.boxplot()
        plt.show()
        print('正态显著性检验', sts.shapiro(sample))                   #Shapiro-Wilk
        print('logistic检验', sts.anderson(sample,dist="logistic"))  #Anderson-Darling

    'Feature Extraction'
    #########################统计指标#######################
    def generateStats(self, data):
        
        def statsTime(oneSample):      #15个时域
            maximum = np.max(oneSample)                                 #最大
            minimum = np.min(oneSample)                                 #最小
            mean = np.mean(oneSample)                                   #均值
            var = np.var(oneSample)                                     #方差
            std = np.std(oneSample)                                     #标准差
            absmean = np.fabs(oneSample).mean()                         #绝对值的均值
            rms = np.sqrt(np.square(oneSample).mean())                  #均方根
            vi = np.sqrt(np.fabs(oneSample)).mean()**2                  #方根幅值
            skew = np.mean((oneSample - mean) ** 3)                     #偏斜度
            kurt = np.mean((oneSample - mean) ** 4) / pow(var, 2)       #峭度
            ptp = np.max(oneSample) - np.min(oneSample)                 #峰峰值
            par  = ptp/rms                                              #峰值因子
            form = rms/absmean                                          #波形因子
            impulse = maximum/absmean                                   #脉冲因子
            margin = maximum/vi                                         #裕度因子
            
            eigenvector = np.array([maximum, minimum, mean, var, std, absmean,
                                    rms, vi, skew, kurt, ptp, par, form, impulse, margin])
            return eigenvector

        def statsFreq(oneSample):       #3个频域
            
            N= oneSample.shape[0]
            mag = np.abs(np.fft.fft(oneSample))           #频谱模
            mag    = mag[1:int(N/2)]*2.00/N
            
            spectral_kurt = sts.kurtosis(mag)             #功率谱峰态
            spectral_skw =  sts.skew(mag)                 #功率谱偏态
            spectral_pow =  np.mean(np.power(mag, 3))     #谱能量
            
            eigenvector = np.array([spectral_kurt, spectral_skw, spectral_pow])
            return eigenvector

        length = len(data)
        eigenvectorSet = []
        for i in range(length):
            eigenvectorSet.append( np.hstack((statsTime(data[i]), statsFreq(data[i]) )) )
        
        return np.array(eigenvectorSet)

    '变换'
    #########################PCA#######################
    def generatePCA(self, data):
        def pca(X):
            # normalize the features
            X = (X - X.mean()) / X.std()
            # compute the covariance matrix
            X = np.matrix(X)
            cov = (X.T * X) / X.shape[0]
            
            # perform SVD
            U, S, V = np.linalg.svd(cov)

            return U, S, V

        def project_data(X, U, k):
            U_reduced = U[:,:k]
            return np.dot(X, U_reduced)
    
        U, S, V = pca(sample)
        Z = project_data(sample, U, 1)
        return Z
    
    ########################小波包变换######################
    def generatePywt(self, data):

        def pywtEigenvector(oneSample):
            #WaveletPacket decomposition
            wp = pywt.WaveletPacket(data=oneSample, wavelet='db1', maxlevel=3, mode='symmetric')
            
            # Min-Max Normalization
            wpnorList = []
            for x in wp['ad'].data:
                x = float(x - np.min(wp['ad'].data))/(np.max(wp['ad'].data)- np.min(wp['ad'].data))
                wpnorList.append(x)
            
            for y in wp['ddd'].data:
                y = float(y - np.min(wp['ddd'].data))/(np.max(wp['ddd'].data)- np.min(wp['ddd'].data))
                wpnorList.append(y)
            
            eigenvector = wpnorList
            return eigenvector

        length = len(data)
        eigenvectorSet = []
        for i in range(length):
            eigenvectorSet.append(pywtEigenvector(data[i]))
        
        return np.array(eigenvectorSet)
    
    #######################手写小波包变换#####################
    def generateMywavelet(data):

        def mywavelet(oneSample):
            wavelet = pywt.Wavelet('db2')          #Daubechies基函数
            Lo_D = np.array(wavelet.dec_lo)        #Decomposition filter
            Hi_D = np.array(wavelet.dec_hi)
            Lo_R = np.array(wavelet.rec_lo)        #Reconstruction filter
            Hi_R = np.array(wavelet.rec_hi)
            print(Hi_D)
            print(Lo_D)
            
            n = 2 * len(Hi_D)
            fx = np.array(oneSample)
            fxL = np.ones(n)*fx.item(0)
            fxR = np.ones(n)*fx.item(len(oneSample)-1)
            fxext = np.hstack((fxL, fx, fxR))
            fxextend = fxext.astype(np.float64)
            
            D1H = np.convolve(fxextend, Hi_D)
            D1L = np.convolve(fxextend, Lo_D)
            
            D1H1 = D1H[n : len(D1H)-n]
            D1H2 = D1H1[n//4 : len(D1H1)-n//4]
            
            D1L1 = D1L[n : len(D1L)-n]
            D1L2 = D1L1[n//4-1 : len(D1L1)-n//4]
            
            D1H2f =scipy.ndimage.zoom(D1H2, 0.5)
            D1L2f =scipy.ndimage.zoom(D1L2, 0.5)
            
            D1H2n = []
            D1L2n = []
            
            for x in D1H2f:
                x = float(x - np.min(D1H2f))/(np.max(D1H2f)- np.min(D1H2f))
                D1H2n.append(x)
            
            for x in D1L2f:
                x = float(x - np.min(D1L2f))/(np.max(D1L2f)- np.min(D1L2f))
                D1L2n.append(x)

            return D1H2n, D1L2n

        length = len(data)
        eigenvectorSet = []
        
        for i in range(length):
            eigenvectorH, eigenvectorL= mywavelet(example[i]) #d a
            eigenvectorHH, eigenvectorHL = mywavelet(eigenvectorH) #dd da
            eigenvectorLH, eigenvectorLL = mywavelet(eigenvectorL) #ad aa
            eigenvectorSet.append(eigenvectorL + eigenvectorHH)

        return np.array(eigenvectorSet)

    #######################小波包系数的L2范数#####################
    def generateWave_L2(self, data):
        
        def wave_fea(a):
            wp = pywt.WaveletPacket(a,'db1', maxlevel=8)      #最大分解水平
            nodes = wp.get_level(8, "freq")   #返回第八级上的 指定波段   256*8
            '矩阵的范数 平方和开根号衡量相应尺度综合程度的数量化指标'
            return np.linalg.norm(np.array([n.data for n in nodes]), 2)

        length = len(data)
        eigenvectorSet = []
        for i in range(length):
            eigenvectorSet.append(wave_fea(data[i]))
        
        return np.array(eigenvectorSet).reshape(length, 1)

    #########################分形维数########################
    def generateFD(self, data):
        
        def refactor(arr, K):     #Subsequence
            N = arr.shape[0]                    #原始序列长度
            curve = []
            for m in range(K):
                sub = arr[m : N : K]            #采样
                Length = sub.shape[0]
                temp_sum = 0
                for i in range(1, Length):
                    temp_sum += abs(sub[i] - sub[i-1])               #分子左侧
                refactor_len = temp_sum * (N-1) / (Length * K * K)   #每个曲线长度
                curve.append(refactor_len)
            
            return np.mean(curve)               #平均长度

        def higuchi_FD(arr, k):    #log变换，线性回归
            temp = [i for i in range(1,k)]
            
            lk = []
            index = []
            for j in temp:
                index.append(np.log10(1/j))
                lk.append(np.log10(refactor(arr, j)))
            
            FD = np.polyfit(index,lk,1)[0]
            return FD, index, lk   #一次线性回归斜率

        length = len(data)
        eigenvectorSet = []
        for i in range(length):
            eigenvectorSet.append(higuchi_FD(data[i], 10)[0])
        
        return np.array(eigenvectorSet).reshape(length, 1)

    #########################弗雷歇距离########################
    def generateFrechet(self, data):
    
        def euc_dist(pt1,pt2):    # Euclidean distance 欧式距离
            return math.sqrt((pt2[0]-pt1[0])*(pt2[0]-pt1[0])+(pt2[1]-pt1[1])*(pt2[1]-pt1[1]))

        def _c(ca,i,j,P,Q):
            if ca[i,j] > -1:
                return ca[i,j]
            elif i == 0 and j == 0:
                ca[i,j] = euc_dist(P[0],Q[0])
            elif i > 0 and j == 0:
                ca[i,j] = max(_c(ca,i-1,0,P,Q),euc_dist(P[i],Q[0]))
            elif i == 0 and j > 0:
                ca[i,j] = max(_c(ca,0,j-1,P,Q),euc_dist(P[0],Q[j]))
            elif i > 0 and j > 0:
                ca[i,j] = max(min(_c(ca,i-1,j,P,Q),_c(ca,i-1,j-1,P,Q),_c(ca,i,j-1,P,Q)),euc_dist(P[i],Q[j]))
            else:
                ca[i,j] = float("inf")
            return ca[i,j]

        """
            Computes the discrete frechet distance between two polygonal lines
            P and Q are arrays of 2-element arrays (points)
            """
        def frechetDist(P,Q):
            ca = np.ones((len(P),len(Q)))
            ca = np.multiply(ca,-1)
            return _c(ca,len(P)-1,len(Q)-1,P,Q)

        def Frechet(data, form):        #与形态的弗雷歇距离
            length = len(data)
            eigenvectorSet = []
            time_index = np.array([i for i in range(2048)])     #时间index
            for i in range(length):
                temp = np.vstack((time_index, data[i]))
                eigenvectorSet.append( frechetDist(temp, form) )
            
            return np.array(eigenvectorSet).reshape(length, 1)

        #八种故障形态
        Frechet_1 = Frechet(data, self.form_1)
        Frechet_2 = Frechet(data, self.form_2)
        Frechet_3 = Frechet(data, self.form_3)
        Frechet_4 = Frechet(data, self.form_4)
        Frechet_5 = Frechet(data, self.form_5)
        Frechet_6 = Frechet(data, self.form_6)
        Frechet_7 = Frechet(data, self.form_7)
        Frechet_8 = Frechet(data, self.form_8)
        
        return np.hstack((Frechet_1, Frechet_2, Frechet_3, Frechet_4, Frechet_5, Frechet_6, Frechet_7, Frechet_8))
    
    '特征分布'
    def featureDistribution(self, feature):
        
        df=pd.DataFrame(feature, columns = ['最大', '最小', '均值', '方差', '标准差', 'absmean', '均方根', 'vi',
        'skew','kurt', 'ptp', 'par', 'form', 'impulse', 'margin', '谱峰态', '谱偏态','谱能量', '波范数',
        'frechet1','frechet2', 'frechet3', 'frechet4', 'frechet5', 'frechet6', 'frechet7', 'frechet8' ])

        nan_rows = df[df.isnull().T.any().T]    #缺失值
        print('特征缺失值数量为', len(nan_rows))
        
        plt.figure(figsize=(16,5))              #离群点
        df.boxplot()
        plt.show()

    '特征变换缩放'
    def featureScaling(self, feature):

#        smooth = np.log1p(feature)                     #logit变换
        min_max_scaler = preprocessing.MinMaxScaler()  #最大最小值归一化
        feature_scaled = min_max_scaler.fit_transform(feature)
#        feature_caled = preprocessing.scale(feature)  #Z-Score标准化
        return feature_scaled

    '特征选择'
    ####################卡方检验######################
    def featureSelect(self, feature, label):
        model1 = SelectKBest(chi2, k=2)                       #卡方检验选择k个最佳特征
        selectfeature = model1.fit_transform(feature, label)  #特征数据与标签数据，可以选择出k个特征
        
        scores = pd.DataFrame(model1.scores_.reshape(1, 27), columns = ['maximum', 'minimum', 'mean', 'var',
                    'std', 'absmean', 'rms', 'vi', 'skew', 'kurt', 'ptp', 'par', 'form', 'impulse', 'margin',
                'spectral_kurt', 'spectral_skw','spectral_pow', 'wave_L2', 'frechet_form1', 'frechet_form2',
        'frechet_form3', 'frechet_form4', 'frechet_form5', 'frechet_form6', 'frechet_form7', 'frechet_form8' ])
        scores.index = ['chi-square']
        scores.T.sort_values(by = 'chi-square', ascending= False)
        
        pvalues = pd.DataFrame(model1.pvalues_.reshape(1,27), columns =['maximum', 'minimum', 'mean', 'var',
                    'std', 'absmean', 'rms', 'vi', 'skew', 'kurt', 'ptp', 'par', 'form', 'impulse', 'margin',
                'spectral_kurt', 'spectral_skw','spectral_pow', 'wave_L2', 'frechet_form1', 'frechet_form2',
        'frechet_form3', 'frechet_form4', 'frechet_form5', 'frechet_form6', 'frechet_form7', 'frechet_form8' ])
        pvalues.index = ['p-value']
        pvalues.T.sort_values(by = 'p-value', ascending= True)

    #########################特征方差########################
    def var_filter(self, data, label, k=0):
        """
        计算dataframe中输入特征方差并按阈值返回dataframe
        :param data: dataframe数据集，包括输入输出
        :param label: 输出特征
        :param k: 方差阈值
        :return:  按阈值返回dataframe
        """
        features = data.drop([label], axis=1).columns
        saved_features = []
        for feature in features:
            feature_var = np.array(data[feature]).var()
            print('输入特征{0}的方差为：{1}'.format(feature, feature_var))

    #########################共线性检验########################
    def vif_test(self, data, label, k=None):
        """
        计算dataframe中输入特征之间的共线性系数
        :param data: dataframe数据集，包括输入输出
        :param label: 输出特征
        :param k: 相关系数阈值
        :return:  按阈值返回dataframe
        """
        features = data.drop([label], axis=1).columns
        feature_array = np.array(data[features])
        #     print(feature_array)
        vif_array = np.corrcoef(feature_array, rowvar=0)
        #     print(vif_array)
        for idx in range(len(features) - 1):
            for count in range(idx + 1, len(features)):
                vif = vif_array[idx][count]
                if vif > k:
                    print('特征{0}与特征{1}的共线性系数vif为：{2}'.format(features[idx], features[count], vif))

    ####################变量预测能力WOE与IV######################
    #等距分箱
    def bin_distince(self, x, y, n=10): # x为待分箱的变量，y为target变量.n为分箱数量
        total = y.count()  # 计算总样本数
        bad = y.sum()      # 计算坏样本数
        good = y.count()-y.sum()  # 计算好样本数
        d1 = pd.DataFrame({'x':x,'y':y,'bucket':pd.cut(x,n)}) #利用pd.cut实现等距分箱
        d2 = d1.groupby('bucket',as_index=True)  # 按照分箱结果进行分组聚合
        d3 = pd.DataFrame(d2.x.min(),columns=['min_bin'])
        d3['min_bin'] = d2.x.min()  # 箱体的左边界
        d3['max_bin'] = d2.x.max()  # 箱体的右边界
        d3['bad'] = d2.y.sum()  # 每个箱体中坏样本的数量
        d3['total'] = d2.y.count() # 每个箱体的总样本数
        d3['bad_rate'] = d3['bad']/d3['total']  # 每个箱体中坏样本所占总样本数的比例
        d3['badattr'] = d3['bad']/bad   # 每个箱体中坏样本所占坏样本总数的比例
        d3['goodattr'] = (d3['total'] - d3['bad'])/good  # 每个箱体中好样本所占好样本总数的比例
        d3['woe'] = np.log(d3['goodattr']/d3['badattr'])  # 计算每个箱体的woe值
        iv = ((d3['goodattr']-d3['badattr'])*d3['woe']).sum()  # 计算变量的iv值
        d4 = (d3.sort_values(by='min_bin')).reset_index(drop=True) # 对箱体从大到小进行排序
        print('分箱结果：')
        print(d4)
        print('IV值为：')
        print(iv)
        cut = []
        cut.append(float('-inf'))
        for i in d4.min_bin:
            cut.append(i)
            cut.append(float('inf'))
            woe = list(d4['woe'].round(3))
            return d4,iv,cut,woe
    
    #等频分箱
    def bin_frequency(self, x, y, n=10): # x为待分箱的变量，y为target变量.n为分箱数量
        total = y.count()  # 计算总样本数
        bad = y.sum()      # 计算坏样本数
        good = y.count()-y.sum()  # 计算好样本数
        d1 = pd.DataFrame({'x':x,'y':y,'bucket':pd.qcut(x,n)})  # 用pd.cut实现等频分箱
        d2 = d1.groupby('bucket',as_index=True)  # 按照分箱结果进行分组聚合
        d3 = pd.DataFrame(d2.x.min(),columns=['min_bin'])
        d3['min_bin'] = d2.x.min()  # 箱体的左边界
        d3['max_bin'] = d2.x.max()  # 箱体的右边界
        d3['bad'] = d2.y.sum()  # 每个箱体中坏样本的数量
        d3['total'] = d2.y.count() # 每个箱体的总样本数
        d3['bad_rate'] = d3['bad']/d3['total']  # 每个箱体中坏样本所占总样本数的比例
        d3['badattr'] = d3['bad']/bad   # 每个箱体中坏样本所占坏样本总数的比例
        d3['goodattr'] = (d3['total'] - d3['bad'])/good  # 每个箱体中好样本所占好样本总数的比例
        d3['woe'] = np.log(d3['goodattr']/d3['badattr'])  # 计算每个箱体的woe值
        iv = ((d3['goodattr']-d3['badattr'])*d3['woe']).sum()  # 计算变量的iv值
        d4 = (d3.sort_values(by='min_bin')).reset_index(drop=True) # 对箱体从大到小进行排序
        print('分箱结果：')
        print(d4)
        print('IV值为：')
        print(iv)
        cut = []
        cut.append(float('-inf'))
        for i in d4.min_bin:
            cut.append(i)
        cut.append(float('inf'))
        woe = list(d4['woe'].round(3))
        return d4,iv,cut,woe

if __name__ == "__main__":

    '导入数据'
    path = '/Users/tung/Python/WorkProject/PHMresearch/WDCNN&LR_FaultDiagnosis/data/0HP'
    preprocess = preprocess()
    x_train, y_train, x_valid, y_valid, x_test, y_test = preprocess.prepro(d_path=path,
                                                                 length=2048,         #2048=5.68小时  周期T=400*10S=66.6min～1h
                                                                 number=200,          #每类样本的数量
                                                                 normal=True,         #是否标准化
                                                                 rate=[0.7,0.2,0.1],  #测试集验证集划分比例
                                                                 enc=True, enc_step=340)
            
    print('训练样本维度:', x_train.shape)
    print(x_train.shape[0], '训练样本个数')
    print('训练标签one-hot维度',y_train.shape[1])
    print('验证样本的维度', x_valid.shape)
    print(x_valid.shape[0], '验证样本个数')
    print('测试样本的维度', x_test.shape)
    print(x_test.shape[0], '测试样本个数')

    '预处理'
    '检查Nan 缺省值'
    test = pd.DataFrame(x_train)
    nan_rows = test[test.isnull().T.any().T]
    print('样本缺失值数量为', len(nan_rows))
    
    FE = featureEngineer()
    'onehot解码'
    y_train_decode = FE.decode(y_train)
    y_valid_decode = FE.decode(y_valid)
    y_test_decode = FE.decode(y_test)

    'compute class weight'
    weights = FE.calssWeight(y_train_decode)
    print('class_weight', weights)
    
    '查看数据分布'
    FE.sampleDistribution(test.T[0])

    '生成特征'
    feature_train = np.hstack((FE.generateStats(x_train), FE.generateWave_L2(x_train), FE.generateFrechet(x_train)))
    feature_valid = np.hstack((FE.generateStats(x_valid), FE.generateWave_L2(x_valid), FE.generateFrechet(x_valid)))
    feature_test = np.hstack((FE.generateStats(x_test), FE.generateWave_L2(x_test), FE.generateFrechet(x_test)))
    print('feature shape', feature_train.shape)

    '查看特征分布'
    FE.featureDistribution(feature_train)
    
    'Feature Scalings缩放'
    feature_train_scaled = FE.featureScaling(feature_train)
    feature_valid_scaled = FE.featureScaling(feature_valid)
    feature_test_scaled = FE.featureScaling(feature_test)

    'Feature Select选择'
    FE.featureSelect(feature_train_scaled, y_train)

    '取前两类做共线性与IV检验数据集'
    class_index = np.argwhere(y_train_decode < 2)
    head = class_index[0][0]
    tail = class_index[-1][0]

    y_train_IV = y_train_decode[head:tail].reshape(tail, 1)
    x_train_IV = feature_train[head:tail]
    x_train_IV = FE.featureScaling(x_train_IV)   #独立归一化

    print('y_train_IV shape {}' .format(y_train_IV.shape))
    print('x_train_IV shape {}' .format(x_train_IV.shape))

    '与标签合并'
    data = np.hstack((x_train_IV, y_train_IV))
    data = pd.DataFrame(data, columns = ['maximum', 'minimum', 'mean', 'var', 'std', 'absmean', 'rms', 'vi',
                'skew', 'kurt', 'ptp', 'par', 'form', 'impulse', 'margin', 'spectral_kurt', 'spectral_skw',
                'spectral_pow', 'wave_L2', 'frechet_form1', 'frechet_form2', 'frechet_form3', 'frechet_form4',
                'frechet_form5', 'frechet_form6', 'frechet_form7', 'frechet_form8', 'y'])

    FE.var_filter(data, 'y', k=0)   #特征方差
    FE.vif_test(data, 'y', k=0.9)   #特征共线性检验

    d4,iv,cut,woe = FE.bin_distince(data['frechet_form8'], data['y'], n =2)    #等距 IV
    d4,iv,cut,woe = FE.bin_frequency(data['frechet_form8'], data['y'], n =3)   #等频 IV

