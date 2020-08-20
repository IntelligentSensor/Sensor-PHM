from __future__ import print_function

from pyspark import SparkContext
# $example on$
from pyspark.mllib.feature import StandardScaler, StandardScalerModel
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.util import MLUtils
# $example off$

if __name__ == "__main__":
    sc = SparkContext(appName="StandardScalerExample")  # SparkContext

    # $example on$
    data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
    label = data.map(lambda x: x.label)
    features = data.map(lambda x: x.features)

    scaler1 = StandardScaler().fit(features)
    scaler2 = StandardScaler(withMean=True, withStd=True).fit(features)

    # data1 will be unit variance.
    data1 = label.zip(scaler1.transform(features))

    # data2 will be unit variance and zero mean.
    data2 = label.zip(scaler2.transform(features.map(lambda x: Vectors.dense(x.toArray()))))
    # $example off$

    print("data1:")
    for each in data1.collect():
        print(each)

    print("data2:")
    for each in data2.collect():
        print(each)

    sc.stop()
