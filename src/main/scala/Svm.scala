import breeze.plot._
import global._

object Main {
  def main(args: Array[String]): Unit = {
    val (df, labels) = Data.readCSV("./res/iris.csv")

    val svm = new Svm(df, labels)


  }
}

object Data {
  def readCSV(path: String): Tuple2[Vector[Vector[Double]], Vector[Int]] = {
    val bufferedSource = scala.io.Source.fromFile(path)

    def flowerClass(v: String): Double = v.trim() match {
      case "virginica" => -1
      case "versicolor" => 1
      case "setosa" => 2
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

object Plotter {
  implicit def toVec[F <: Iterable[_]](x: F): Vector[_] = x.toVector

  def apply[T: Numeric, F[_] <: Iterable[_]](x: DataFrame[Double], labels: F[T]): Unit = {
    val f = Figure()
    val p = f.subplot(0)

    val x1 = (x, labels).zipped.filter((_, b) => b == -1)._1
    val x2 = (x, labels).zipped.filter((_, b) => b == 1)._1

    p += plot(x1.map(_.head), x1.map(_.last), '.', "b")
    p += plot(x2.map(_.head), x2.map(_.last), '.', "r")
////    p += plot(List( 5.0, 8.0), List(w(0) * 5, w(0) * 8))
    f.saveas("t.png")
  }
}

//https://stats.stackexchange.com/questions/5056/comp1-0.000uting-the-decision-boundary-of-a-linear-svm-model
class Svm(x: DataFrame[Double], labels: Vector[Int], eta: Double=1, epochs: Int=10000) {

  val df: DataFrame[Double] = x.map(_ :+ 1.0)

  // weights initialization
  var w: Vector[Double] = 0.to(df(0).length).map(i => i * 0.0).toVector


  def fit(): Unit = {
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
  fit()

  println("Weigths", w)

  println("Classification accuracy", (classification(df, w).map(_.signum), labels).zipped.count(i => i._1 == i._2).toDouble / x.length)
  Plotter[Int, Vector](x, labels)


  // Misclassification treshold
  def misClassification(x: Vector[Double], w: Vector[Double], label: Int): Boolean = {
    dotProduct(x, w) * label < 1
  }

  def classification(x: Vector[Vector[Double]], w: Vector[Double]): Vector[Double] = x.map(dotProduct(_, w))

  def dotProduct(x: Vector[Double], w: Vector[Double]): Double = (x, w).zipped.map((a, b) => a * b).sum


  def updateWeightsIfWrong(w: Vector[Double], data: Vector[Double], label: Int, epoch: Int): Vector[Double] = {
    (w, data).zipped.map((w, d) => w + eta * ((d * label) + (-2 * (1 / epoch) * w)))
  }

  def regularization(w: Vector[Double], label: Int, epoch: Int): Vector[Double] = {
    w.map(i => i + eta * (-2  * (1 / epoch) * i))
  }

}

