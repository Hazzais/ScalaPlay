package logisticregression

import breeze.linalg._
import breeze.numerics.{exp, log}
import breeze.stats.mean

import scala.annotation.tailrec


class LogisticRegression(learningRate: Double, maxIterations: Int) {

  private var modelWeights: DenseVector[Double] = _

  private def addBiasColumn(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    // Add a column to feature matrix corresponding to bias (e.g. all ones)
    val fBias = DenseMatrix.ones[Double](x.rows, 1)
    val featuresFull = DenseMatrix.horzcat(fBias, x)
    featuresFull
  }

  private def sigmoid(z: DenseVector[Double]): DenseVector[Double] = {
    // Sigmoid function: compresses number in range [0, 1]
    1d /:/ (1d +:+ exp(-z))
  }

  private def gradientDescent(x: DenseMatrix[Double], y: DenseVector[Double], sigma: DenseVector[Double]): DenseVector[Double] = {
    // Find amount weights should change using gradient descent
    val nRows: Double = y.length
    (x.t * (sigma -:- y)) / nRows
  }

  private def findLoss(y: DenseVector[Double], sigma: DenseVector[Double]): Double = {
    // Calculate loss (smaller is better)
    mean(-y *:* log(sigma) -:- (1d -:- y) *:* log(1d -:- sigma))
  }

  @tailrec
  private def iterateFit(nIterations: Int, x: DenseMatrix[Double],
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

  def fit(trainX: DenseMatrix[Double], trainY: DenseVector[Double]): Unit = {
    // Fit logistic regression to training data

    // Need intercept column to be added
    val featuresFull = addBiasColumn(trainX)

    // Initialise weights as all zero
    val nFeatures = featuresFull.cols
    val weights = DenseVector.zeros[Double](nFeatures)

    // Iteratively, find optimal weights
    this.modelWeights = iterateFit(this.maxIterations, featuresFull, trainY, weights, this.learningRate)
  }

  def predictProba(x: DenseMatrix[Double]): DenseVector[Double] = {
    // Predict probabilities for set of data
    val featuresFull = addBiasColumn(x)
    sigmoid(featuresFull * this.modelWeights)
  }

}
