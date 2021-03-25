package logisticregression

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.numerics.{exp, log}
import breeze.linalg._
import breeze.stats.mean
import breeze.linalg.csvread

import java.io.File
import scala.annotation.tailrec

// TODO: a lot of the logistic regression functions could/should be in a class. A few can be made private.

object LogisticRegression {

  def addBiasColumn(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    // Add a column to feature matrix corresponding to bias (e.g. all ones)
    val fBias = DenseMatrix.ones[Double](x.rows, 1)
    val featuresFull = DenseMatrix.horzcat(fBias, x)
    featuresFull
  }

  def sigmoid(z: DenseVector[Double]): DenseVector[Double] = {
    // Sigmoid function: compresses number in range [0, 1]
    1d /:/ (1d +:+ exp(-z))
  }

  def gradientDescent(x: DenseMatrix[Double], y: DenseVector[Double], sigma: DenseVector[Double]): DenseVector[Double] = {
    // Find amount weights should change using gradient descent
    val nRows: Double = y.length
    (x.t * (sigma -:- y)) / nRows
  }

  def findLoss(y: DenseVector[Double], sigma: DenseVector[Double]): Double = {
    // Calculate loss (smaller is better)
    mean(-y *:* log(sigma) -:- (1d -:- y) *:* log(1d -:- sigma))
  }

  @tailrec
  def iterateFit(nIterations: Int, x: DenseMatrix[Double],
                 y: DenseVector[Double], weights: DenseVector[Double], learningRate: Double): DenseVector[Double] = {
    // Recursively update weights according to gradient descent until desired number of iterations is complete.
    // Could have used a loop for simplicity but wanted to try out tail recursion (not that it would have a big impact
    // here).

    // TODO: some return structure to also include loss (appendable) and any other thing needed
    // TODO: add early stopping if loss doesn't change above a threshold

    // Algorithm is:
    // 1) Multiply features and weights
    // 2) Find what the output of the sigmoid function is (0 to 1)
    // 3) Change weight slightly so as to reduce loss between predicted probability and labels
    // 4) Repeat until max number of iterations complete (convex loss function means if convergence reached, minima will
    //    not be exited so long as learning rate not too high. Ideally, would like to automatically break out if
    //    converged.
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

  def logisticRegressionFit(trainX: DenseMatrix[Double], trainY: DenseVector[Double],
                            maxIterations: Int, learningRate: Double): DenseVector[Double] = {
    // Fit logistic regression to training data

    // Need intercept column to be added
    val featuresFull = addBiasColumn(trainX)

    // Initialise weights as all zero
    val nFeatures = featuresFull.cols
    val weights = DenseVector.zeros[Double](nFeatures)

    // Iteratively, find optimal weights
    iterateFit(maxIterations, featuresFull, trainY, weights, learningRate)
  }

  def logisticRegressionPredProba(x: DenseMatrix[Double],
                                  fittedWeights: DenseVector[Double]): DenseVector[Double] = {
    // Predict probabilities for set of data
    val featuresFull = addBiasColumn(x)
    sigmoid(featuresFull * fittedWeights)
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
    1 - mean((predLabels - trueLabels))
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

    // Fit model
    val fittedWeights = logisticRegressionFit(trainX, trainY, maxIterations=10, learningRate=0.5)

    // Predict on train set
    val trainProbs = logisticRegressionPredProba(trainX, fittedWeights)
    val trainPreds = thresholdProbability(trainProbs)
    val trainAccuracy = accuracyScore(trainPreds, trainY)
    println(s"Training accuracy: $trainAccuracy")

    // Predict on test set
    val testProbs = logisticRegressionPredProba(testX, fittedWeights)
    val testPreds = thresholdProbability(testProbs)
    val testAccuracy = accuracyScore(testPreds, testY)
    println(s"Testing accuracy: $testAccuracy")
  }
}
