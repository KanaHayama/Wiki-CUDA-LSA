package ee451s2019

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Matrix, SingularValueDecomposition, Vectors}

object MLlibSVD extends SVD {
	def get(sparkSession: SparkSession, documentTermFrequencyMatrix: DocumentTermFrequencyMatrix, k: Int) : SingularValueDecomposition[RowMatrix, Matrix] = {
		val rdd = documentTermFrequencyMatrix.idfMatrix.select(DocumentTermFrequencyMatrix.TERM_IDF_COL_NAME).rdd.map { row =>
			Vectors.fromML(row.getAs[Vector](DocumentTermFrequencyMatrix.TERM_IDF_COL_NAME))
		}
		rdd.cache()
		val matrix = new RowMatrix(rdd)
		matrix.computeSVD(k, computeU = true)
	}
}
