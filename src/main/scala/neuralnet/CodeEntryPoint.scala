package neuralnet


object CodeEntryPoint {
  def calculate(a: Int, b: Int): Int = {
    a + b
  }

  def main(args: Array[String]): Unit = {
    import breeze.linalg.csvread
    import java.io.File
    val matrix=csvread(new File("iris.csv"),',', skipLines = 1)
//    val result = calculate(1, 2)
//    println(s"Result is ${result}")
  }
}

/*
Misc things:
- Random splitter
- Standardiser

Network:
- Dense


 */

//object CodeEntryPoint {
//  def calculate(a: Int, b: Int): Int = {
//    a + b
//  }
//
//  def main(args: Array[String]): Unit = {
//    val result = calculate(1, 2)
//    println(s"Result is ${result}")
//  }
//}
