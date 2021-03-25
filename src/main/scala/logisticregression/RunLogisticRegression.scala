package logisticregression

import breeze.linalg.csvread

import java.io.File


object RunLogisticRegression {

  def main(args: Array[String]): Unit = {
    // Example usage of LogisticRegression
    // Takes a CSV (filename specified in arg(0)), and builds logistic regression model to use all but final column, to
    // predict 0 or 1 of final column. Does simple train/test split.
    // Assumptions:
    //  - Final column in CSV is what needs to be modelled and is only 0 and 1 (rows with column equal or greater than
    //    two are removed though)
    //  - All feature columns numerics
    //  - All feature columns are roughly the same scale (no scaling performed). Otherwise model will be poor.
    //  - No regularistion needs to occur (should be a simple amendment to loss/gradient descent function if needed)
    //  - Learning rate is 0.5, number of iterations is 20. Another couple of arguments would need to be added to allow
    //    the user to change this, but should be quick to do.

    // Parse input arguments
    val filepath = args(0)
    val stripHeader = if (args(1) == "true") 1 else 0
    val testSize = args(2).toDouble

    // Getting data together
    val matrix = csvread(new File(filepath), ',', skipLines = stripHeader)
    val features = matrix(::, (0 to (matrix.cols - 2)).toList)
    val labels = matrix(::, -1)

    // Select only two classes (0 and 1)
    val wantedLabels = labels(labels <:< 2.0).toDenseVector
    val wantedFeatures = features(labels <:< 2.0, ::).toDenseMatrix

    // Make hold out set
    val (trainX, testX, trainY, testY) = train_test_split(wantedFeatures, wantedLabels, test_size = testSize)

    // Fit model
    val model = new LogisticRegression(learningRate = 0.5, maxIterations = 20)
    model.fit(trainX, trainY)

    // Predict on train set
    val trainProbs = model.predictProba(trainX)
    val trainPreds = thresholdProbability(trainProbs)
    val trainAccuracy = accuracyScore(trainPreds, trainY)
    println(s"Training accuracy: $trainAccuracy")

    // Predict on test set
    val testProbs = model.predictProba(testX)
    val testPreds = thresholdProbability(testProbs)
    val testAccuracy = accuracyScore(testPreds, testY)
    println(s"Testing accuracy: $testAccuracy")
  }
}
