"""
    Logistic regression using MLlib.
    
    This example requires NumPy (http://www.numpy.org/).
    """
from __future__ import print_function

import sys

from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD


def parsePoint(line):
    """
        Parse a line of text into an MLlib LabeledPoint object.
        """
    values = [float(s) for s in line.split(' ')]
    if values[0] == -1:   # Convert -1 labels to 0 for MLlib
        values[0] = 0
    return LabeledPoint(values[0], values[1:])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: logistic_regression <file> <iterations>", file=sys.stderr)
        sys.exit(-1)
    sc = SparkContext(appName="PythonLR")
    points = sc.textFile(sys.argv[1]).map(parsePoint)
    iterations = int(sys.argv[2])
    model = LogisticRegressionWithSGD.train(points, iterations)
    print("Final weights: " + str(model.weights))
    print("Final intercept: " + str(model.intercept))
    sc.stop()
