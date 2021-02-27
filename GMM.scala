
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import scala.math._

/**
 *  Univariate Gaussian Mixture Model - Expectation Maximization
 *  Erick Escobar Gallardo -  2021
 */

object GMM  {
  type GMM = (Array[Double],Array[Double], Array[Double])
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    Logger.getLogger("org.spark-project").setLevel(Level.ERROR)
    val conf = new SparkConf().setAppName("Erick Escobar - GMM").setMaster("local[*]")
    //val conf = new SparkConf().setAppName("Erick Escobar - GMM")
    val sc = new SparkContext(conf)
    val X: RDD[Double] = sc.textFile("C:/Users/erick/Downloads/Spark_test/dataset/dataset-mini.txt") //load the dataset
    //val X: RDD[Double] = sc.textFile("/data/bigDataSecret/dataset-big.txt") //load the dataset
      .map(r => r.toDouble).persist() // cast the dataset values to double and persist it.
    // Defining parameters of the EM model
    val epsilon: Double = 10000 // define the value of epsilon
    val K: Int = 2 // defines K value
    val maxCycles: Int = 100 // define the maximum number of cycles, if EM calculation is taking a lot of time, this will ensure the stopping process of the cycles.
    val numCyclesLnp: Int = 10 // define the number of cycles to wait in order to calculate the log joint probability error
    // Returning and printing final Weights, Means and Variances
    val startTime = System.currentTimeMillis()
    val ( mean,variance,weights )= EM(X,K,epsilon,maxCycles,numCyclesLnp)
    val endTime = System.currentTimeMillis()
    println("_______________Final values_______________")
    println("K:",K)
    println("weights_final: ")
    weights.foreach(println)
    println("means_final: ")
    mean.foreach(println)
    println("variances_final: ")
    variance.foreach(println)
    println("Time: ", endTime-startTime)
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
    val s1: Double = (1.0 / math.sqrt(2.0*Pi*variance))
    val s2: Double = math.exp(-math.pow((x-mean),2.0)/ (2.0*variance))
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
    for (k<-0 until K)
    {
      partial = partial + weights(k) * pdf(x, variance(k), mean(k))
    }
    partial
  }



  def EM(X:RDD[Double], K:Int, epsilon:Double,maxCycles:Int,numCyclesLnp:Int): GMM = {

    /**
     * Step: Initialization
     * In this step we will initializate the weights,means, and variances and calculate the total count of the dataset.
     */


    val totalCount: Long = X.count() //obtain the total number of members in the dataset
    val variance: Double = X.variance() //obtain the variance of the dataset
    val datasetVariance: Array[Double] = Array.fill(K)(variance) // fill a K-array with the variance of the dataset
    val datasetWeights: Array[Double] = Array.fill(K)(1.0 / K) // fill a K-array with the initial weights
    //val datasetMean: Array[Double] = X.takeSample(withReplacement = true, K, System.nanoTime.toInt)  //obtain K samples from the dataset and use them as the means values
    val datasetMean: Array[Double] = X.takeSample(withReplacement = true, K, 3)  //obtain K samples from the dataset and use them as the means values
    /**
     * logJointProbability calculates the log of the joint probability
     *
     * @return the log joint probability
     */

    def logJointProbability = (acc: Double, value: Double) => acc + math.log(kSumLogLikelihood(K, value, datasetWeights, datasetVariance, datasetMean)) // calculates the logLikelihood

    var lnp: Double = X.aggregate(0.0)(logJointProbability, _+_) // calculates the loglikelihood for the initial values (means,weights,variances)
    var cLnp: Double = 0.0
    var count:Int = 0 // initialize counter to 0
    var cycles:Int = 0 // initialize counter to 0
    var updateTriggered = false // flag to calculate log joint probability each 10 iterations
    var continue = true // flag to control stop condition
    println("_______________Initial values_______________")
    println("Size of dataset: ", totalCount)
    println("Mean: ")
    datasetMean.foreach(println)
    println("Weights: ")
    datasetWeights.foreach(println)
    println("Variance ")
    datasetVariance.foreach(println)
    println("Lnp: ", lnp)
    println("Epsilon:", epsilon)
    do {
      /**
       * Step: Expectation
       */

      // Gamma:RDD[Array[Double]] will contain K+1 elements, the first K elements contain the likelihood (ynk) the last element contains the data point from the X:RDD[double]
      val gamma: RDD[Array[Double]] = X.map(r => {
        val tempArray: Array[Double] = Array.fill(K + 1)(0.0) // Temp Array of size K +1
        val divisor: Double = kSumLogLikelihood(K, r, datasetWeights, datasetVariance, datasetMean) // Sum of the logLikelihood for K values, we calculate this outside of the loop
        for (k <- 0 to K - 1) {
          tempArray(k) = (datasetWeights(k) * pdf(r, datasetVariance(k), datasetMean(k))) / divisor
        }
        tempArray(K) = r
        tempArray
      }).persist() // need to persist since the gamma RDD will be used to calculate the new weights,variances and means

      /**
       * Step: Maximization
       * The following lines of codes perform the Maximization step
       */
      for (k <- 0 to K - 1) {
        val sumLikelihoods = gamma.aggregate(0.0)((acc: Double, v: Array[Double]) => acc + v(k), _ + _) // Calculates the sum of the gamma values for k value
        // Update datasetWeights array.
        datasetWeights(k) = sumLikelihoods / totalCount  // update the weights
        // Update datasetMean array.
        datasetMean(k) = (gamma.aggregate(0.0)((acc: Double, v: Array[Double]) => acc + v(k) * v(K), _ + _)) / sumLikelihoods  // update the means
        // Update datasetVariance array.
        datasetVariance(k) = (gamma.aggregate(0.0)((acc: Double, v: Array[Double]) => acc + v(k) * pow((v(K) - datasetMean(k)), 2), _ + _)) / sumLikelihoods //update the variances
      }

      /**
       * Step: Convergence Measurement
       */

      if (count == numCyclesLnp || count == 0) { // update the log joint probability each numCyclesLnp
        cLnp = lnp //store previous log joint probability
        println("update triggered!")
        updateTriggered = true
        lnp = X.aggregate(0.0)(logJointProbability, _ + _) // calculate the new log likelihood value
        count = 0
      }
      // if - else conditional to inform about updates in mean, variance and weights each cycle
      if (updateTriggered) {
        println(cycles, lnp, cLnp, "difference: ", abs(lnp - cLnp), "mean:", datasetMean.mkString(" , "), "variance:", datasetVariance.mkString(" , "), "weights:", datasetWeights.mkString(" , "))
        updateTriggered = false // reset flag
      }
      else{
        println(cycles, "mean:", datasetMean.mkString(" , "), "variance:", datasetVariance.mkString(" , "), "weights:", datasetWeights.mkString(" , "))
      }
      // If conditional to control exit condition 1: the differences between log joint probabilities have to below an epsilon value
      if ( abs(lnp - cLnp) < epsilon){
        println("control 1")
        println("final lnp:")
        println(abs(lnp))
        continue = false // set flag to false, exit loop
      }
      // If conditional to control exit condition 2: the total number of cycles has been reached
      if( cycles == maxCycles){
        println("control 2")
        println("final lnp:")
        println(abs(lnp))
        continue = false // set flag to false, exit loop
      }
      count += 1
      cycles += 1
      gamma.unpersist() // un-persist gamma RDD to prevent OOM related errors
    //} while ((abs(lnp - cLnp) > epsilon) || (cycles < maxCycles))
    } while (continue)
     (datasetMean, datasetVariance, datasetWeights)
  }

}

