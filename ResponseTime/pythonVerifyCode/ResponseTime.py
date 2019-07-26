#-*- coding: UTF-8 -*-

import math
import numpy as np
import pandas as pd

def getincrease(array):
    begin = 0
    maxLen = 1
    k = 1               #递增子序列长度
    stable = 6.7
    result = []
    
    for i in range(1, len(array)):
        if array[i] > array[i-1]:
            k +=1
        else:
            k=1
        if k >= maxLen:
            maxLen = k
            begin = i - maxLen + 1
            
    threshold = abs(array[begin+maxLen] - stable)
    
    if maxLen>8 and threshold < 0.2:
        for j in range(begin, begin+maxLen):
            result.append(array[j])  
        
    print("样品液响应时间:%10.0f" %maxLen)    
    return result      

def getdecrease(array):
    begin = 0
    maxLen = 1
    k = 1              #递减子序列长度
    stable = 3.5
    result = []
    
    for i in range(1, len(array)):
        if array[i] < array[i-1]:
            k +=1
        else:
            k=1
        if k >= maxLen:
            maxLen = k
            begin = i - maxLen + 1
            
    threshold = abs(array[begin+maxLen] - stable)
    
    if maxLen > 8 and threshold < 0.2:
        for j in range(begin, begin+maxLen):
            result.append(array[j])   
            
    print("缓冲液响应时间:%10.0f" %maxLen)  
    return result

def readfile():
    scale = 50
    file_name = 'PH 0019-05-21.txt'
    data = []
    for line in open(file_name , encoding='gbk' , errors='ignore'):
        line = line.split()
        data.append(line)   

    frame = pd.DataFrame(data , columns=['time','a','b','ph','temp'])
    frame.index.name = 'no'

    frameph = frame.ph.apply(lambda x: x[1:5]) #delete string pH

    frameph = [float(i) for i in frameph]
    frameph = pd.DataFrame(frameph, columns=['ph'])

    frameph = frameph.iloc[::-1]                        #列反转
    frameph = frameph.reset_index(drop=True)   #重建索引从零开始
    frameph.insert(0,'time',frameph.index)
   
    cut = frameph.ph[460:820]
    phdiff = np.diff(cut, n=1, axis = -1)
    phdiff2 = np.diff(cut, n=2, axis = -1)

    median = cut.rolling(window=29).median()     #中值滤波 非线性平滑 保留信号边缘
    median = list(median)
    
def plot():
    plt.figure(figsize=(5,3))

    plt.plot(getincrease(median)) 
    pl.legend(loc='upper left')
    plt.title("HK-328 1s")

    plt.plot(getdecrease(median)) 
    pl.legend(loc='upper left')
    plt.title("HK-328 1s")
    