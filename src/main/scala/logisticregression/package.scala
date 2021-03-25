import breeze.linalg.{DenseMatrix, DenseVector}

package object logisticregression {

  def train_test_split(x: DenseMatrix[Double], y: DenseVector[Double],
                       test_size: Double):
  (DenseMatrix[Double], DenseMatrix[Double], DenseVector[Double], DenseVector[Double]) = {

    val randNumbers = DenseVector.rand(x.rows)

    val trainRows =  randNumbers >:> test_size
    val testRows =  !trainRows

    val trainX = x(trainRows, ::).toDenseMatrix
    val trainY = y(trainRows).toDenseVector

    val testX = x(testRows, ::).toDenseMatrix
    val testY = y(testRows).toDenseVector
    (trainX, testX, trainY, testY)
  }

}
