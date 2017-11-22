import breeze.plot._
import global._

object Main {
  def main(args: Array[String]): Unit = {
    val (df, labels) = Data.readCSV("./res/iris.csv")

    val svm = new Svm(df, labels)
    svm.fit()
    println("Weigths", svm.w)
    println("Classification accuracy", (svm.classification(df).map(_.signum), labels).zipped.count(i => i._1 == i._2).toDouble / svm.df.length)

    Plotter(svm.df, labels, svm.w)

  }
}

object Data {
  def readCSV(path: String): Tuple2[Vector[Vector[Double]], Vector[Int]] = {
    val bufferedSource = scala.io.Source.fromFile(path)

    def flowerClass(v: String): Double = v.trim() match {
      case "setosa" => -1
      case "versicolor" => 1
      case "virginica" => 2
      case x => x.toDouble
    }
    val lines = bufferedSource.getLines.toVector.tail
    val rows = lines.map(_.split(",").toVector.map(flowerClass)).filter(i => i.last < 2)
    val labels = rows.map(_.last.toInt)
    val data = rows.map(_.init)
    bufferedSource.close()

    (data, labels)
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

class Svm(x: DataFrame[Double], labels: Vector[Int], eta: Double=1, epochs: Int=10000) {

  // Take 2 features
  val df: DataFrame[Double] = x.map(i => Vector(i(0), i(1), 1))

  // weights initialization
  var w: Vector[Double] = 0.to(df(0).length).map(i => i * 0.0).toVector


  def fit(): Unit = {
    def updateWeightsIfWrong(w: Vector[Double], data: Vector[Double], label: Int, epoch: Int): Vector[Double] = {
      (w, data).zipped.map((w, d) => w + eta * ((d * label) + (-2 * (1 / epoch) * w)))
    }

    // Misclassification treshold
    def misClassification(x: Vector[Double], w: Vector[Double], label: Int): Boolean = {
      dotProduct(x, w) * label < 1
    }

    def regularization(w: Vector[Double], label: Int, epoch: Int): Vector[Double] = {
      w.map(i => i + eta * (-2  * (1 / epoch) * i))
    }

    def trainOneEpoch(w: Vector[Double], x: DataFrame[Double], labels: Vector[Int], epoch: Int): Vector[Double] = (x, labels) match {
      case (xh +: xs, lh +: ls) if misClassification(xh, w, lh) => trainOneEpoch(updateWeightsIfWrong(w, xh, lh, epoch), xs, ls, epoch)
      case (_ +: xs, lh +: ls) => trainOneEpoch(regularization(w, lh, epoch), xs, ls, epoch) // if correct update regularization parameter
      case _ => w
    }

    def trainEpochs(w: Vector[Double], epochs: Int, epochCount: Int = 1): Vector[Double] = epochs match {
      case 0 => w
      case _ => trainEpochs(trainOneEpoch(w, df, labels, epochCount), epochs - 1, epochCount + 1)
    }

    w = trainEpochs(w, epochs)

  }

  def classification(x: Vector[Vector[Double]], w: Vector[Double] = w): Vector[Double] = x.map(dotProduct(_, w))

}

