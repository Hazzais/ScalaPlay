package logisticregression
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.numerics.{exp, log}
import breeze.linalg._
import breeze.stats.mean
import breeze.linalg.csvread
import java.io.File

//class LogisticRegression() {
//
//  def sigmoid(input)
//
//
//}

//https://stackoverflow.com/questions/38935272/how-can-i-write-logistic-regression-with-scala-breeze-with-lbfgs?rq=1

object RunLogisticRegression {


  def sigmoid(z: DenseVector[Double]): DenseVector[Double] = {
    1d /:/ (1d +:+ exp(-z))
  }

  def gradientDescent(x: DenseMatrix[Double], y: DenseVector[Double], sigma: DenseVector[Double]): DenseVector[Double] = {
    val nRows: Double = y.length
    (x.t * (sigma -:- y)) / nRows
  }

//  def loss(labels: DenseVector[Double], sigma: DenseVector[Double]): Double = {
//    mean(-labels *:* log(sigma) -:- (1 -:- labels) *:* log(1 -:- sigma))
//  }

  def iterateFit(nIterations: Int, x: DenseMatrix[Double],
                 y: DenseVector[Double], weights: DenseVector[Double], learningRate: Double): DenseVector[Double] = {

    // TODO: some return structure to also include loss (appendable) and any other thing needed
    // TODO: add early stopping if loss doesn't change above a threshold

    val z = x * weights
    val sigma = sigmoid(z)
    val loss = mean(-y *:* log(sigma) -:- (1d -:- y) *:* log(1d -:- sigma))
    println(s"Loss: $loss")

    val deltaWeights = gradientDescent(x, y, sigma)
    weights :-= (learningRate  *:* deltaWeights)

    if (nIterations < 1) {
      weights
    } else {
      iterateFit(nIterations - 1, x, y, weights, learningRate)
    }
  }

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

  def main(args: Array[String]): Unit = {

    // Getting data together
    val matrix = csvread(new File("iris.csv"), ',', skipLines = 1)
    val features = matrix(::, (0 to (matrix.cols - 2)).toList)
    val labels = matrix(::, -1)

    // Select only two classes (0 and 1)
    val wantedLabels = labels(labels <:< 2.0).toDenseVector
    val wantedFeatures = features(labels <:< 2.0, ::).toDenseMatrix

    // Make hold out set
    val (trainX, testX, trainY, testY) = train_test_split(wantedFeatures, wantedLabels, test_size = 0.25)

    val fBias = DenseMatrix.ones[Double](trainX.rows, 1)
    val featuresFull = DenseMatrix.horzcat(fBias, trainX)

    val nFeatures = featuresFull.cols
    val weights = DenseVector.zeros[Double](nFeatures)
    val maxIterations = 20
    val learningRate = 0.5

    val finalWeights = iterateFit(maxIterations, featuresFull, trainY, weights, learningRate)

    val preProb = sigmoid(featuresFull * finalWeights)

    val predictions = preProb
//    val predsCond = preProb <:< 0.5
    predictions(preProb <:< 0.5) := 0.0
    predictions(preProb >:> 0.5) := 1.0

    val accuracy = 1 - mean((predictions - trainY))
    println(s"Training accuracy: $accuracy")
  }
}
