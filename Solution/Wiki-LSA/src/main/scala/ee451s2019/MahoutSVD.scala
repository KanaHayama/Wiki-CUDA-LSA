package ee451s2019

import org.apache.spark.ml.linalg.{Vector => MLVector}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.mahout.math.Vector
import org.apache.mahout.math.drm._
import org.apache.mahout.math.decompositions.DSSVD
import org.apache.mahout.sparkbindings.drmWrapMLLibVector
import org.apache.spark.sql.SparkSession

class MahoutSVD(val u: DrmLike[Int], val s: Vector, val v: DrmLike[Int]) {

}

object MahoutSVD {
	def get(sparkSession: SparkSession, documentTermFrequencyMatrix: DocumentTermFrequencyMatrix, k: Int) : MahoutSVD = {
		val rdd = documentTermFrequencyMatrix.idfMatrix.select(DocumentTermFrequencyMatrix.TERM_IDF_COL_NAME).rdd.map { row =>
			Vectors.fromML(row.getAs[MLVector](DocumentTermFrequencyMatrix.TERM_IDF_COL_NAME))
		} //TODO: not optimal
		val drm = drmWrapMLLibVector(rdd)
		val (drmU, drmV, s) = DSSVD.dssvd(drm, k)
		new MahoutSVD(drmU, s, drmV)
	}
}
