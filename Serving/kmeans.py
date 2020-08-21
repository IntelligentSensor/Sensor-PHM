#!/usr/bin/env python

import os
import sys
from datetime import datetime
from sklearn import preprocessing
from pyspark import SparkContext
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.clustering import KMeans
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.stat import MultivariateStatisticalSummary

#启动spark
spark_path = os.environ.get('SPARK_HOME', None)
sys.path.insert(0, os.path.join(spark_path, 'python/lib/py4j-0.10.7-src.zip'))
exec(open(os.path.join(spark_path, 'python/pyspark/shell.py')).read())

#创建RDD
total_outlier_sample = sc.parallelize(total_outlier_sample)
print (total_fault_sample.take(1))

#数据的相关因素特征向量的分布,决定是否归一化
'转为分布式行矩阵'
total_outlier_matrix = RowMatrix(total_outlier_sample)

'生成DenseVectorde的概述统计量'
desc_total_outlier_matrix = MultivariateStatisticalSummary(total_outlier_matrix.rows)

# print ('outlier factors mean:',desc_total_outlier_matrix.mean())
# print ('outlier factors variance:',desc_total_outlier_matrix.variance())

#训练
num_clusters = 9
num_iterations = 20
num_runs =3

outlier_cluster_model = KMeans.train(total_outlier_sample,num_clusters, num_iterations, num_runs)
outlier_predictions = outlier_cluster_model.predict(total_outlier_sample)

print ('对前十个outlier样本的预测标签为:'+",".join([str(i) for i in outlier_predictions.take(10)]))

#评估模型
#内部指标WCSS
outlier_cost = outlier_cluster_model.computeCost(total_outlier_sample)
print ("WCSS for outlier_sample: %f"%outlier_cost)

#外部指标
#带标注的数据 分类指标

#交叉验证K调优
train_test_split_outlier = total_outlier_sample.randomSplit([0.6,0.4],123)
train_outlier = train_test_split_outlier[0]
test_outlier = train_test_split_outlier[1]
for k in [2,3,4,5,6,7,8,10,12,14,16,18,20,22,24,26,28,30]:
    k_model = KMeans.train(train_outlier, num_iterations, k, num_runs)
    cost = k_model.computeCost(test_outlier)
    print ('WCSS for k=%d : %f'%(k,cost))

'手写demo'

#归一化与反归一化
Scaler = preprocessing.MinMaxScaler()
Scaler_data = Scaler.fit_transform(total_outlier_sample)
origin_data = Scaler.inverse_transform(Scaler_data)

sc.stop()

sc = SparkContext('local[2]', appName='outlier_KMeans')
lines = sc.parallelize(Scaler_data)
data = lines.cache()       #数据缓存加速

#这个函数的目的就是把读入的数据都转化为float类型的数据
def parseVector(line):
    return np.array([float(x) for x in line.split(' ')])

#求取该点应该分到哪个点集中去，返回的是序号
def closestPoint(p, centers):
    bestIndex = 0
    closest = float("+inf")
    for i in range(len(centers)):
        tempDist = np.sum((p - centers[i]) ** 2)   #欧式距离
        if tempDist < closest:
            closest = tempDist
            bestIndex = i
    return bestIndex

if len(sys.argv) != 4:
    print("Usage: kmeans <file> <k> <convergeDist>", file=sys.stderr)
    exit(-1)

print("""WARN: This is a naive implementation of KMeans Clustering and is given
    as an example! Please refer to examples/src/main/python/mllib/kmeans.py for an example on
    how to use MLlib's KMeans implementation.""", file=sys.stderr)

#K是设置的中心点数
K = int(9)
#设置的阈值，如果两次之间的距离小于该阈值的话则停止迭代
convergeDist = float(0.01)
#从点集中用采样的方式来抽取K个值
kPoints = data.takeSample(False, K, 1)
print('聚类中心shape', np.shape(kPoints))

##对所有数据执行map过程，最终生成的是(index, (point, 1))的rdd
#closest = data.map(lambda p: (closestPoint(p, kPoints), (p, 1)))
#input_ = closest.take(1)
#print(input_[0][0])
#
##执行reduce过程，该过程的目的是重新求取中心点，生成的也是rdd
#pointStats = closest.reduceByKey(lambda p1_c1, p2_c2: (p1_c1[0] + p2_c2[0], p1_c1[1] + p2_c2[1]))
#temp = pointStats.take(1)
#print(temp[0][0])
#
##生成新的中心点
#newPoints = pointStats.map(lambda st: (st[0], st[1][0] / st[1][1])).collect()
#newPoints[0]

start = datetime.now()
#中心点调整后的距离差
tempDist = 1.0

#如果新旧中心点的距离差大于阈值则执行
while tempDist > convergeDist:
    #对所有数据执行map过程，最终生成的是(index, (point, 1))的rdd
    closest = data.map(lambda p: (closestPoint(p, kPoints), (p, 1)))
    #执行reduce过程，该过程的目的是重新求取中心点，生成的也是rdd
    pointStats = closest.reduceByKey(lambda p1_c1, p2_c2: (p1_c1[0] + p2_c2[0], p1_c1[1] + p2_c2[1]))
    #生成新的中心点
    newPoints = pointStats.map(lambda st: (st[0], st[1][0] / st[1][1])).collect()
    #计算一下新旧中心点的距离差
    tempDist = sum(np.sum((kPoints[iK] - p) ** 2) for (iK, p) in newPoints)
    
    #设置新的中心点
    for (iK, p) in newPoints:
       kPoints[iK] = p

print("Final centers: " + str(kPoints))
print("This took ", datetime.now() - start)
sc.stop()

#可视化异常聚类中心
plt.figure(figsize=(15,8))

plt.subplot(3,3,1)
plt.plot(kPoints[0])
plt.title('outlier 4073 samples kPoints 0')
plt.xlabel('时间/min', fontdict={'family' : 'SimHei', 'size'   : 12})
plt.ylabel('pH', fontdict={'family' : 'SimHei', 'size'   : 12})

plt.subplot(3,3,2)
plt.plot(kPoints[1])
plt.title('kPoints 1')
plt.subplot(3,3,3)
plt.plot(kPoints[2])
plt.title('kPoints 2')

plt.subplot(3,3,4)
plt.plot(kPoints[3])
plt.title('kPoints 3')
plt.subplot(3,3,5)
plt.plot(kPoints[4])
plt.title('kPoints 4')
plt.xlabel('时间/min', fontdict={'family' : 'SimHei', 'size'   : 12})
plt.ylabel('pH', fontdict={'family' : 'SimHei', 'size'   : 12})
plt.subplot(3,3,6)
plt.plot(kPoints[5])
plt.title('kPoints 5')


plt.subplot(3,3,7)
plt.plot(kPoints[6])
plt.title('kPoints 6')
plt.subplot(3,3,8)
plt.plot(kPoints[7])
plt.title('kPoints 7')
plt.subplot(3,3,9)
plt.plot(kPoints[8])
plt.title('kPoints 8')
plt.xlabel('时间/min', fontdict={'family' : 'SimHei', 'size'   : 12})
plt.ylabel('pH', fontdict={'family' : 'SimHei', 'size'   : 12})

plt.tick_params(labelsize=12)
plt.tight_layout()
