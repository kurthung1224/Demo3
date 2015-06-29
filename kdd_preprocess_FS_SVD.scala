import org.apache.spark.rdd.RDD
val rawData = sc.textFile("/kdd/kddcup.data_10_percent_corrected")


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
  buffer.clear
  (key)
}
val protocol_temp_keys = protocol_temp.countByValue.keys.toArray
val protocol = one_hot_encoding(protocol_temp_keys)


//obtain mapping values of "service"
val service_temp = rawData.map{ line=>
  val buffer = line.split(',').toBuffer
  buffer.remove(3,buffer.length-3)
  val key = buffer.remove(2)
  buffer.clear
  (key)
}
val service_temp_keys = service_temp.countByValue.keys.toArray
val service = one_hot_encoding(service_temp_keys)


//obtain mapping values of "flag"
val flag_temp = rawData.map{ line=>
  val buffer = line.split(',').toBuffer
  buffer.remove(4,buffer.length-4)
  val key = buffer.remove(3)
  buffer.clear
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


val protocol_location = 1
val protocol_insert_location = protocol_location + 1
val protocol_size = protocol.keys.size


val service_location = protocol_location + protocol_size 
val service_insert_location = service_location + 1
val service_size = service.keys.size


val flag_location = service_location + service_size
val flag_insert_location = flag_location + 1



val rawVector = rawData.map { line =>
  val buffer = line.split(',').toBuffer
  
  buffer.insertAll(protocol_insert_location, protocol(buffer(protocol_location)))
  buffer.remove(protocol_location,1)

  buffer.insertAll(service_insert_location, service(buffer(service_location)))
  buffer.remove(service_location,1)

  buffer.insertAll(flag_insert_location, flag(buffer(flag_location)))
  buffer.remove(flag_location,1) 

  val vector = buffer.init.map(_.toDouble).toArray
  buffer.clear
  (vector)
}


//feature scaling -- start
import scala.collection.mutable.ArrayBuffer

def transpose_Array(k:Array[Array[Double]]): Array[Array[Double]] = {
  val row_size = k.size  //2
  val col_size = k(0).length  //3
  var zz = ArrayBuffer[Double]()
  var ll = ArrayBuffer[Array[Double]]()
  for(j <- 0 until col_size){
    for(i <- 0 until row_size){
      zz += k(i)(j)
    }
	val zz_Array = zz.toArray
	zz.clear
	ll += zz_Array
  }
  val jj = ll.toArray
  ll.clear
  (jj)
}


val rawVector_Array = rawVector.take(rawVector.count.toInt) //This array would occupy numerous mem space
val trans_rawVector_Array = transpose_Array(rawVector_Array)


def minMax(a: Array[Double]) : (Double, Double) = {
  if (a.isEmpty) throw new java.lang.UnsupportedOperationException("array is empty")
    a.foldLeft((a(0), a(0)))
  { case ((min, max), e) => (math.min(min, e), math.max(max, e))}
}

def FS(a:Double, min:Double, max:Double) : (Double) = {
  if(min == max){ //prevent NaN
    (a)
  }else{
    val frac = max - min
    val div = a - min
    val new_v = (div / frac)
    (new_v)
  }
}

val features_num = trans_rawVector_Array.size
var tmp = ArrayBuffer[(Double, Double)]()
for(i <- 0 until features_num){
  tmp += minMax(trans_rawVector_Array(i))
}
val compared_array = tmp.toArray
tmp.clear


val data = rawData.map { line =>
  val buffer = line.split(',').toBuffer
  
  buffer.insertAll(protocol_insert_location, protocol(buffer(protocol_location)))
  buffer.remove(protocol_location,1)

  buffer.insertAll(service_insert_location, service(buffer(service_location)))
  buffer.remove(service_location,1)

  buffer.insertAll(flag_insert_location, flag(buffer(flag_location)))
  buffer.remove(flag_location,1) 

  for(i <- 0 until features_num){
	val tmp_num = FS(buffer(i).toDouble, compared_array(i)._1, compared_array(i)._2)
	buffer.insert(i+1, tmp_num.toString)
	buffer.remove(i,1) 
  }
  
  val vector = Vectors.dense(buffer.init.map(_.toDouble).toArray)
  
  buffer.remove(0, buffer.length - 1)
  buffer.insert(1, signal(buffer(0)).toString)
  buffer.remove(0)

  val label = buffer(0).toDouble
  
  (label, vector)
}
//feature scaling -- end


//Create a label point
val labelsAndData = data.map{point =>
  LabeledPoint(point._1, point._2)
}

//save data to HDFS
import org.apache.spark.mllib.util._
MLUtils.saveAsLibSVMFile(labelsAndData, "hdfs://master01:9000/kdd_output/FS_LablePoint")


//SVD ==>
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.SingularValueDecomposition

val mat = new RowMatrix(data.values)

//dimension reduction to 3-dim or 2-dim
val svd = mat.computeSVD(3, computeU = true)
val U = svd.U
val s = svd.s

//Vector to diag matrix
val diag_s = Matrices.diag(s)

//comput the reduced matrix
val reduce_matrix = U.multiply(diag_s)

//save the matrix
reduce_matrix.rows.saveAsTextFile("/kdd/3_dim")
//SVD <==
