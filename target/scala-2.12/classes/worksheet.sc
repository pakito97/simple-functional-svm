val w = Vector(0.0, 0.0, 0.0, 0.0, 0.0)
val x = Vector(7.0, 3.2, 4.7, 1.4)

val (y +: ys) = x

ys


def test(a: Vector[Double]): Double = a match {
  case y +: ys => test(ys) + y
  case _ => 0.0
}

