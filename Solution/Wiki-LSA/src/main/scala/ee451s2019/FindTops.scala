package ee451s2019

import org.apache.spark.mllib.linalg.{Matrix, SingularValueDecomposition}
import org.apache.spark.mllib.linalg.distributed.RowMatrix

import scala.collection.Map
import scala.collection.mutable.ArrayBuffer

class FindTops(private val svd: SingularValueDecomposition[RowMatrix, Matrix], private val docIds: Map[Long, String], private val termIds:Array[String]) {

	def topTermsInTopConcepts(numConcepts: Int, numTerms: Int): Seq[Seq[(String, Double)]] = {
		val topTerms = new ArrayBuffer[Seq[(String, Double)]]()
		val vArray = svd.V.toArray
		for (i <- 0 until numConcepts) {
			val offset = i * svd.V.numRows
			val termWeights = vArray.slice(offset, offset + svd.V.numRows).zipWithIndex
			val sorted = termWeights.sortBy(-_._1)
			topTerms += sorted.take(numTerms).map { case (weight, id) => (termIds(id), weight) }
		}
		topTerms
	}

	def topDocsInTopConcepts(numConcepts: Int, numDocs: Int): Seq[Seq[(String, Double)]] = {
		val topDocs = new ArrayBuffer[Seq[(String, Double)]]()
		for (i <- 0 until numConcepts) {
			val docWeights = svd.U.rows.map(_.toArray(i)).zipWithUniqueId
			topDocs += docWeights.top(numDocs).map { case (weight, id) => (docIds(id), weight) }
		}
		topDocs
	}
}
