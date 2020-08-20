#!/usr/bin/env python
"""
    Linear Regression With SGD Example.
    """
from __future__ import print_function

import os
import sys
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel

if __name__ == "__main__":
    #启动spark
    spark_path = os.environ.get('SPARK_HOME', None)
    sys.path.insert(0, os.path.join(spark_path, 'python/lib/py4j-0.10.7-src.zip'))
    exec(open(os.path.join(spark_path, 'python/pyspark/shell.py')).read())

    sc.stop()
    sc = SparkContext(appName="PythonLinearRegressionWithSGDExample")
    
    # Load and parse the data
    def parsePoint(line):
        values = [float(x) for x in line.replace(',', ' ').split(' ')]
        return LabeledPoint(values[0], values[1:])
    
    data = sc.textFile("data/mllib/ridge-data/lpsa.data")
    parsedData = data.map(parsePoint)
    
    # Build the model
    model = LinearRegressionWithSGD.train(parsedData, iterations=100, step=0.00000001)
    
    # Evaluate the model on training data
    valuesAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
    MSE = valuesAndPreds \
        .map(lambda vp: (vp[0] - vp[1])**2) \
        .reduce(lambda x, y: x + y) / valuesAndPreds.count()
    print("Mean Squared Error = " + str(MSE))

# Save and load model
model.save(sc, "target/tmp/pythonLinearRegressionWithSGDModel")
sameModel = LinearRegressionModel.load(sc, "target/tmp/pythonLinearRegressionWithSGDModel")
