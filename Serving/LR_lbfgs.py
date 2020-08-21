from __future__ import print_function

from pyspark import SparkContext
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint

if __name__ == "__main__":
    
    sc = SparkContext(appName="PythonLogisticRegressionWithLBFGSExample")
    
    # $example on$
    # Load and parse the data
    def parsePoint(line):
        values = [float(x) for x in line.split(' ')]
        return LabeledPoint(values[0], values[1:])
    
    data = sc.textFile("/Users/tung/Documents/spark-2.4.3/data/mllib/sample_svm_data.txt")
    parsedData = data.map(parsePoint)
    
    # Build the model
    model = LogisticRegressionWithLBFGS.train(parsedData)
    
    # Evaluating the model on training data
    labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
    trainErr = labelsAndPreds.filter(lambda lp: lp[0] != lp[1]).count() / float(parsedData.count())
    print("Training Error = " + str(trainErr))
    
    # Save and load model
#    model.save(sc, "target/tmp/pythonLogisticRegressionWithLBFGSModel")
#    sameModel = LogisticRegressionModel.load(sc,
#                                             "target/tmp/pythonLogisticRegressionWithLBFGSModel")
# $example off$

