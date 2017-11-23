import breeze.plot._
import global._

object Main {
  def main(args: Array[String]): Unit = {
    // load data
    val (df, labels) = readCSV("./res/iris.csv")

    // initialize new SVM object
    val svm = new SVM(df, labels)
    // train svm
    svm.fit()

    println("Classification accuracy:", (svm.classification(svm.df), labels).zipped.count(i => i._1 == i._2).toDouble / svm.df.length)
    println("Weigths:", svm.w)

    Plotter(svm.df, labels, svm.w)

  }
}

object readCSV {
  /**
    * Reads the iris dataset and return the 'setosa' and 'versicolor' class and takes the first two feature columns.
    *
    * @param path Path of the iris.csv file
    * @return a dataframe containing the features and the labels.
    */
  def apply(path: String): (DataFrame[Double], Vector[Int]) = {
    val bufferedSource = scala.io.Source.fromFile(path)

    // rename the setosa and versicolor class to -1 and 1
    def flowerClass(v: String): Double = v.trim() match {
      case "setosa" => -1
      case "versicolor" => 1
      case "virginica" => 10
      case x => x.toDouble
    }

    // read lines
    val lines = bufferedSource.getLines.toVector.tail

    // split lines and map the flowerClass function discarding the virginica flower.
    val rows = lines.map(_.split(",").map(flowerClass)).filter(i => i.last < 2)
    val labels = rows.map(_.last.toInt)
    val data = rows.map(_.init)
    bufferedSource.close()

    (data.map(i => Vector(i(0), i(1))), labels)
  }
}

object dotProduct {
  def apply(x: Vector[Double], w: Vector[Double]): Double = (x, w).zipped.map((a, b) => a * b).sum
}

object Plotter {

  def apply(x: DataFrame[Double], labels: Vector[Int], weights: Vector[Double]): Unit = {
    val f = Figure()
    val p = f.subplot(0)

    val x1 = (x, labels).zipped.filter((_, b) => b == -1)._1
    val x2 = (x, labels).zipped.filter((_, b) => b == 1)._1

    p += plot(x1.map(_(0)), x1.map(_(1)), '.', "b")
    p += plot(x2.map(_(0)), x2.map(_(1)), '.', "r")

    val sorted_x = x.sortWith((i1, i2) => i1(0) > i2(0))
    val w_ = weights.patch(1, Vector(0.0), 1)
    p += plot(sorted_x.map(_(0)), sorted_x.map(x => -dotProduct(x, w_) /  weights(1)))
    f.saveas("t.png")
  }
}

/**
  * @param x Feature data.
  * @param labels Binary labels
  * @param eta Learning rate
  * @param epochs No. of training epochs
  */
class SVM(x: DataFrame[Double], labels: Vector[Int], eta: Double=1, epochs: Int=10000) {

  // Add a bias term to the data.
  def prepare(x: DataFrame[Double]): DataFrame[Double] = x.map(_ :+ 1.0)

  // Prepared data.
  val df: DataFrame[Double] = prepare(x)


  // Weights initialization.
  var w :Vector[Double] = (for (_ <- 1 to df(0).length) yield 0.0).toVector


  def fit(): Unit = {
    // Will only be called if classification is wrong.
    def gradient(w: Vector[Double], data: Vector[Double], label: Int, epoch: Int): Vector[Double] = {
      (w, data).zipped.map((w, d) => w + eta * ((d * label) + (-2 * (1 / epoch) * w)))
    }

    def regularizationGradient(w: Vector[Double], label: Int, epoch: Int): Vector[Double] = {
      w.map(i => i + eta * (-2  * (1 / epoch) * i))
    }

    // Misclassification treshold.
    def misClassification(x: Vector[Double], w: Vector[Double], label: Int): Boolean = {
      dotProduct(x, w) * label < 1
    }

    def trainOneEpoch(w: Vector[Double], x: DataFrame[Double], labels: Vector[Int], epoch: Int): Vector[Double] = (x, labels) match {
        // If classification is wrong. Update weights with loss gradient
      case (xh +: xs, lh +: ls) if misClassification(xh, w, lh) => trainOneEpoch(gradient(w, xh, lh, epoch), xs, ls, epoch)
        // If classification is correct: update weights with regularizer gradient
      case (_ +: xs, lh +: ls) => trainOneEpoch(regularizationGradient(w, lh, epoch), xs, ls, epoch)
      case _ => w
    }

    def trainEpochs(w: Vector[Double], epochs: Int, epochCount: Int = 1): Vector[Double] = epochs match {
      case 0 => w
      case _ => trainEpochs(trainOneEpoch(w, df, labels, epochCount), epochs - 1, epochCount + 1)
    }

    // Update weights
    w = trainEpochs(w, epochs)
  }

  def classification(x: Vector[Vector[Double]], w: Vector[Double] = w): Vector[Int] = x.map(dotProduct(_, w).signum)
}

