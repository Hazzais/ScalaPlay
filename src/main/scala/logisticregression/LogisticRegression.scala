package logisticregression

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.numerics.{exp, log}
import breeze.linalg._
import breeze.stats.mean
import breeze.linalg.csvread
import java.io.File

object RunLogisticRegression {

  def sigmoid(z: DenseVector[Double]): DenseVector[Double] = {
    1d /:/ (1d +:+ exp(-z))
  }

  def gradientDescent(x: DenseMatrix[Double], y: DenseVector[Double], sigma: DenseVector[Double]): DenseVector[Double] = {
    val nRows: Double = y.length
    (x.t * (sigma -:- y)) / nRows
  }

  def findLoss(y: DenseVector[Double], sigma: DenseVector[Double]): Double = {
    mean(-y *:* log(sigma) -:- (1d -:- y) *:* log(1d -:- sigma))
  }

  def iterateFit(nIterations: Int, x: DenseMatrix[Double],
                 y: DenseVector[Double], weights: DenseVector[Double], learningRate: Double): DenseVector[Double] = {

    // TODO: some return structure to also include loss (appendable) and any other thing needed
    // TODO: add early stopping if loss doesn't change above a threshold

    val z = x * weights
    val sigma = sigmoid(z)
    val loss = findLoss(y, sigma)
    println(s"Loss: $loss")

    val deltaWeights = gradientDescent(x, y, sigma)
    weights :-= (learningRate  *:* deltaWeights)

    if (nIterations < 1) {
      weights
    } else {
      iterateFit(nIterations - 1, x, y, weights, learningRate)
    }
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
