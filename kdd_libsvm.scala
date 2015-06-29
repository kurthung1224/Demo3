bin/spark-shell --jars "/tmp/kdd/spark-liblinear-1.95.jar"


import org.apache.spark.rdd.RDD
import tw.edu.ntu.csie.liblinear._

val SVM_data = Utils.loadLibSVMData(sc, "hdfs://master01:9000/kdd_output/part-*")

val model = SparkLiblinear.train(SVM_data, "-s 0 -c 1.0")

def PredictAndRecall(dataPoint:RDD[DataPoint] ,trainedModel: LiblinearModel, labelsNum:Double): (Double, Double) = { 
  val tp = dataPoint.filter{point => (point.y == labelsNum) && (point.y == trainedModel.predict(point))}.count.toDouble
  val fp = dataPoint.filter{point => (point.y == labelsNum) && (point.y != trainedModel.predict(point))}.count.toDouble
  val tn = dataPoint.filter{point => (point.y != labelsNum) && (point.y == trainedModel.predict(point))}.count.toDouble
  val fn = dataPoint.filter{point => (point.y != labelsNum) && (point.y != trainedModel.predict(point))}.count.toDouble
  val pos_prec = tp / (tp+fp)
  val pos_rec  = tp / (tp+fn)
  (pos_prec, pos_rec)
}

//Log result
val x = new Array[(Double, Double)](model.label.size)
for(i <- 0 until model.label.size)
  x(i) = PredictAndRecall(SVM_data, model , model.label(i))
val result = x.zipWithIndex

val signal_swap = signal.map(_.swap)
val sorted_result = result.sortWith(_._1._1 > _._1._1)

//result function
def result(): Unit = {
  for(i <- 0 until model.label.size)
    if(sorted_result(i)._1._1 != 0.0 &&  sorted_result(i)._1._2 != 0.0)
      println(signal_swap(sorted_result(i)._2) + " = " + sorted_result(i)._1)
}

//print result
result
