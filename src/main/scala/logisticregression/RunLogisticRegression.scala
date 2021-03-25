package logisticregression

import breeze.linalg.csvread

import java.io.File


object RunLogisticRegression {

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
    val model = new LogisticRegression(0.5, 20)
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
