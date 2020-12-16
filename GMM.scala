
import au.com.bytecode.opencsv.CSVParser
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import scala.math._
//import org.apache.spark.{RangePartitioner, SparkConf, SparkContext}

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
    EM(X,K)
    System.in.read()
  }

  /**
   * Calculate the probability density function: Gaussian form
    * @param x : data point
   * @param variance : variance of the dataset
   * @param mean : mean of the dataset
   * @return the probability density
   */
  def pdf( x:Double,variance:Double, mean:Double):Double = {
    val s1: Double = (1.0 / sqrt(2.0*Pi*variance))
    val s2: Double = exp(-pow(x-mean,2)/ (2.0*variance))
    s1*s2
  }

  /**
   *  Calculates the sum of all likelihoods of a given x point (for all K clusters)
   * @param K : K clusters
   * @param x : data point
   * @param weights : weights array
   * @param variance : variance array
   * @param mean: mean array
   * @return the sum of all the likelihoods
   */
  def kSumLogLikelihood(K:Int, x:Double, weights:Array[Double], variance:Array[Double], mean:Array[Double]):Double ={
    var partial:Double = 0.0
    for (k<-0 to K-1)
      {
        partial += weights(k)*pdf(x,sqrt(variance(k)),mean(k))
      }
    partial
  }


  def EM(X:RDD[Double], K:Int): Unit = {

    /**
     * Step: Initialization
     */

    val epsilon:Double = 1e-8
    val totalCount: Long = X.count()
    val mean: Double = X.mean()
    val variance: Double = X.variance()
    val datasetVariance: Array[Double] = Array.fill(K)(variance)
    val datasetWeights: Array[Double] = Array.fill(K)(1.0/K)
    val test_datasetWeights: Array[Double] = Array.fill(K)(0.0) //test variable
    val datasetMean: Array[Double] = X.takeSample(withReplacement = true, K, seed = 3)
    val test_datasetMean: Array[Double] = Array.fill(K)(0.0)  //test variable
    //val datasetMean: Array[Double] = X.takeSample(withReplacement = true, K, System.nanoTime.toInt)  // Un-comment later, for now we want fixed samples.
    /**
     * logJointProbability calculates the log of the joint probability
     * @return the log joint probability
     */
    def logJointProbability = (acc:Double, v:Double) => {
      var partial: Double = 0.0
      partial = kSumLogLikelihood(K,v,datasetWeights,datasetVariance,datasetMean)
      acc + log(partial)
    }
    var lnp: Double = X.aggregate(0.0)(logJointProbability,_+_)

    /**
     * Step: Expectation
     */

    // Gamma:RDD[Array[Double]] will contain K+1 elements, the first K elements contain the likelihood (ynk) the last element contains the data point from the X:RDD[double]
    val gamma: RDD[Array[Double]] = X.map(r => {
      val tempArray:Array[Double] = Array.fill(K+1)(0.0) // Temp Array of size k
      val divisor:Double = kSumLogLikelihood(K, r, datasetWeights, datasetVariance, datasetMean) // Sum of the loglikelihood for each value of K
      for (k <- 0 to K - 1) {
        tempArray(k) = (datasetWeights(k) * pdf(r, sqrt(datasetVariance(k)), datasetMean(k))) / divisor
      }
      tempArray(K) = r
      tempArray
    }) // hard implementation, need to persist? persist here causes to mantain the same values.
    var test = gamma.collect().take(1).toList

    /**
    * Step: Maximization
    */

    for (k<-0 to K-1) {
      // Update datasetWeights array.
      val sumLikelihoods = gamma.aggregate(0.0)((acc:Double,v:Array[Double])=> acc + v(k),_+_)
      //test_datasetWeights(k) = (gamma.aggregate(0.0)((acc:Double,v:Array[Double])=> acc + v(k),_+_))/ totalCount
      test_datasetWeights(k) = sumLikelihoods / totalCount
      // Update datasetMean array.
      //test_datasetMean(k) = gamma.aggregate((0.0))
      // Update datasetVariance array.
    }








    //Exporting Gamma matrix
    //System.setProperty("hadoop.home.dir", "C:/hadoop")
    //val export = gamma.map(r => Array(r(0),r(1),r(2)).mkString(",")  )
    //export.saveAsTextFile("dataset/omega.csv")
    //X.saveAsTextFile("dataset/testdata.txt")
    // Step: Maximization

//    do {
//
//    } while (true)

    //Debugging prints
    println(X.toDebugString)
    println("total count: ",totalCount)
    println("dataset mean: ", mean)
    println("dataset variance: ", variance)
    println("Array: Samples for mean: ")
    datasetMean.foreach(println)
    println("Array: Initial weights: ")
    datasetWeights.foreach(println)
    println("Array:: dataset variance ")
    datasetVariance.foreach(println)
    println("logtest: ", lnp)
    println("epsilon:", epsilon)
    println("weights_iter(1): ")
    test_datasetWeights.foreach(println)
    //println(test)
    //print("testi: ",testi)
    //sampler(X,totalCount,k)
    //test.foreach(println)

    test.foreach(a => println(a(0),a(1),a(2),a(3))) //gamma data
  }



}

