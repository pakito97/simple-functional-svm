import breeze.plot._

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



class Svm(x: Vector[Vector[Double]], labels: Vector[Int], eta: Double = 1, epochs: Int = 10000) {

  // weights initialization
  var w: Vector[Double] = 0.to(x(0).length).map(i => i * 0.0).toVector
  val currentepoch: Int = 1


  def fit(): Unit = {

    def trainOneEpoch(w: Vector[Double], x: Vector[Vector[Double]], labels: Vector[Int], epoch: Int): Vector[Double] = (x, labels) match {
      case (xh +: xs, lh +: ls) if misClassification(xh, w, lh) => trainOneEpoch(updateWeightsIfWrong(w, xh, lh, epoch), xs, ls, epoch)
      case (_ +: xs, lh +: ls) => trainOneEpoch(regularization(w, lh, epoch), xs, ls, epoch) // if correct update regularization parameter
      case _ => w
    }

    def trainEpochs(w: Vector[Double], epochs: Int, epochCount: Int = 1): Vector[Double] = epochs match {
      case 0 => w
      case _ => trainEpochs(trainOneEpoch(w, x, labels, epochCount), epochs - 1, epochCount + 1)
    }

    w = trainEpochs(w, epochs)

  }
  fit()

  println(w)
  println((classification(x, w).map(_.signum), labels).zipped.count(i => i._1 == i._2).toDouble / x.length)
  val f = Figure()
  val p = f.subplot(0)
  p += plot(x.map(_.head), x.map(_.last), '.')
  p += plot(List(w(0) * 5, w(0) * 8), List(w.last * 1, w.last*3))
  f.saveas("t.png")

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

