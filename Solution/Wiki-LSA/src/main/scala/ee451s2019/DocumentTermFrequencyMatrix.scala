package ee451s2019

import org.apache.spark.sql.DataFrame

class DocumentTermFrequencyMatrix(val idfMatrix: DataFrame, val docIds: Map[Long, String], val termIds: Array[String], val idfScale: Array[Double]) extends Serializable {

}

object DocumentTermFrequencyMatrix {
	final val TITLE_COL_NAME = DocumentTermFrequency.TERMS_COL_NAME
	final val TERM_IDF_COL_NAME = DocumentTermFrequency.TERM_IDF_COL_NAME
}