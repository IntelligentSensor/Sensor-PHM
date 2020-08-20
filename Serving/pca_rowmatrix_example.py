from pyspark import SparkContext
# $example on$
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix
# $example off$

if __name__ == "__main__":
    sc = SparkContext(appName="PythonPCAOnRowMatrixExample")

    # $example on$
    rows = sc.parallelize([
        Vectors.sparse(5, {1: 1.0, 3: 7.0}),
        Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
        Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
    ])

    mat = RowMatrix(rows)
    # Compute the top 4 principal components.
    # Principal components are stored in a local dense matrix.
    pc = mat.computePrincipalComponents(4)

    # Project the rows to the linear space spanned by the top 4 principal components.
    projected = mat.multiply(pc)
    # $example off$
    collected = projected.rows.collect()
    print("Projected Row Matrix of principal component:")
    for vector in collected:
        print(vector)
    sc.stop()
