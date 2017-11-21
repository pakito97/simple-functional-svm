val w = Vector(0.0, 0.0, 0.0, 0.0, 0.0)
val x = Vector(7.0, 3.2, 4.7, 1.4, 5.0)

def testFunc[T, V <: Seq[T]](x: V) = x

testFunc[Double, Seq[Double]](List(1, 3))
//testFunc(Vector(1, 3))



