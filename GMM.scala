//import breeze.numerics.{exp, pow, sqrt}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.rand
import scala.math._
import org.apache.spark.{RangePartitioner, SparkConf, SparkContext}
object GMM extends App {
  type GMM = (Array[Double],Array[Double], Array[Double])
  override def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    Logger.getLogger("org.spark-project").setLevel(Level.ERROR)
    val conf = new SparkConf().setAppName("Erick Escobar - GMM").setMaster("local[2]")
    val sc = new SparkContext(conf)
    // Reading dataset
    val X: RDD[Double] = sc.textFile("dataset/dataset-sample.txt")
      .map(r => r.split(" "))
      .map(r => r(0).toDouble).persist()

    val K: Int = 3
    //val test: Double = X.collect().take(1)(0)
    //val testi: Double = pdf(test,variance,mean)
    EM(X,K)



    System.in.read()
  }

  def pdf( x:Double,variance:Double, mean:Double):Double = {
    val s1: Double = (1.0 / sqrt(2.0*Pi*variance))
    val s2: Double = exp(-pow(x-mean,2)/ (2.0*variance))
    return s1*s2
  }


  def EM(X:RDD[Double], K:Int): Unit = {
    // Variables
    val totalCount: Long = X.count()
    val mean: Double = X.mean()
    val variance: Double = X.variance()
    val datasetVariance: Array[Double] = Array.fill(K)(variance)
    val datasetWeights: Array[Double] = Array.fill(K)(1.0/K)
    val datasetMean: Array[Double] = X.takeSample(withReplacement = true, K, seed = 3)
    //val datasetMean: Array[Double] = X.takeSample(withReplacement = true, K, System.nanoTime.toInt)  // Un-comment later, for now we want fixed samples.
    def logLikelihood = (accu:Double, v:Double) => {
      var partial: Double = 0.0
      for (k <- 0 to K-1)
        {
          partial += datasetWeights(k)*pdf(v,sqrt(datasetVariance(k)),datasetMean(k))
        }
      accu + log(partial)
    }
    def operation2 = (accu1:Double, accu2:Double) => accu1 + accu2
    val lnp: Double = X.aggregate(0.0)(logLikelihood,_+_)

    //Debugging
    println(X.toDebugString)
    println(totalCount)
    println("dataset mean: ", mean)
    println("dataset variance: ", variance)
    datasetMean.foreach(println)
    datasetWeights.foreach(println)
    datasetVariance.foreach(println)
    println("logtest: ", lnp)
    //println(test)
    //print("testi: ",testi)
    //sampler(X,totalCount,k)

  }



}

