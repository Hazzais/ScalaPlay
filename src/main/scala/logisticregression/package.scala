import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.mean

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

  def thresholdProbability(predProbs: DenseVector[Double], threshold: Double = 0.5): DenseVector[Double] = {
    // Assign labels based on whether probabilities are above or below a threshold
    val predsAbove = predProbs >:> threshold

    val predsThresholded = DenseVector.ones[Double](predProbs.length)
    predsThresholded(predsAbove) := 1.0
    predsThresholded(!predsAbove) := 0.0

    predsThresholded
  }

  def accuracyScore(predLabels: DenseVector[Double], trueLabels: DenseVector[Double]): Double = {
    // Simple accuracy score
    1 - mean(predLabels - trueLabels)
  }
}
