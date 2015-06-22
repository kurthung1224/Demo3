import org.apache.spark.rdd.RDD

val rawData = sc.textFile("/kdd/kddcup.data_10_percent_corrected")

val data = rawData.map(_.split(','))


//one-hot encoding
def one_hot_encoding(word:Array[String]): Map[String, Array[String]] = {
  var hot = new Array[String](word.length)
  var kk = new Array[Array[String]](word.length)
  var states = collection.immutable.Map[String, Array[String]]()
  for(i <- 0 until word.length){
    for(j <- 0 until word.length){
      if(j==i) hot(j) = 1.toString else hot(j) = 0.toString
    }
    val x = hot.reverse
    kk(i) = x
    states += (word(i) -> kk(i))
  }
  (states)
}



//obtain mapping values of "protocol"
val protocol_temp = rawData.map{ line=>
  val buffer = line.split(',').toBuffer
  buffer.remove(2,buffer.length-2)
  val key = buffer.remove(1)
  (key)
}
val protocol_temp_keys = protocol_temp.countByValue.keys.toArray
val protocol = one_hot_encoding(protocol_temp_keys)



//obtain mapping values of "service"
val service_temp = rawData.map{ line=>
  val buffer = line.split(',').toBuffer
  buffer.remove(3,buffer.length-3)
  val key = buffer.remove(2)
  (key)
}
val service_temp_keys = service_temp.countByValue.keys.toArray
val service = one_hot_encoding(service_temp_keys)



//obtain mapping values of "flag"
val flag_temp = rawData.map{ line=>
  val buffer = line.split(',').toBuffer
  buffer.remove(4,buffer.length-4)
  val key = buffer.remove(3)
  (key)
}
val flag_keys_temp = flag_temp.countByValue.keys.toArray
val flag = one_hot_encoding(flag_keys_temp)



//obtain mapping values of "signal"
val signal_temp = rawData.map{ line=>
  val buffer = line.split(',').toBuffer
  buffer.remove(0, buffer.length - 1)
  (buffer(0))
}
val signal_keys_temp = signal_temp.countByValue.keys.toArray
val signal = signal_keys_temp.zipWithIndex.toMap


import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._


//catergorial to number
val labelsAndData = rawData.map { line =>
  val buffer = line.split(',').toBuffer
  
  buffer.insertAll(2, protocol(buffer(1)))
  buffer.remove(1,1)

  buffer.insertAll(5, service(buffer(4)))
  buffer.remove(4,1)

  buffer.insertAll(71, flag(buffer(70)))
  buffer.remove(70,1) 

  val vector = Vectors.dense(buffer.init.map(_.toDouble).toArray)

  buffer.remove(0, buffer.length - 1)
  buffer.insert(1, signal(buffer(0)).toString)
  buffer.remove(0)

  val label = buffer(0).toDouble

  (label, vector)
}

//Normalize data ==>
val labelsAndData_values = labelsAndData.values

val dataAsArray = labelsAndData_tmp.map(_.toArray)
val numCols = dataAsArray.first().length
val n = dataAsArray.count()
  
val sums = dataAsArray.reduce(
  (a,b) => a.zip(b).map(t => t._1 + t._2)
)

val sumSquares = dataAsArray.fold(
  new Array[Double](numCols)
)(
  (a,b) => a.zip(b).map(t => t._1 + t._2 * t._2)
)

val stdevs = sumSquares.zip(sums).map {
  case(sumSq,sum) => math.sqrt(n*sumSq - sum*sum)/n
}

val means = sums.map(_ / n)

def normalize(datum: Vector) = {
  val normalizedArray = (datum.toArray, means, stdevs).zipped.map(
    (value, mean, stdev) =>
      if (stdev <= 0) (value - mean) else (value - mean) / stdev
  )
  Vectors.dense(normalizedArray)
}
//Normalize data <==


val z = labelsAndData.map{point =>
  LabeledPoint(point._1, normalize(point._2))
}



//save data to HDFS
import org.apache.spark.mllib.util._
MLUtils.saveAsLibSVMFile(z, "hdfs://master01:9000/kdd/norm_data")