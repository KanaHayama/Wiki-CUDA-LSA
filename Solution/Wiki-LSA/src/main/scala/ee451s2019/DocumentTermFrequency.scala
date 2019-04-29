package ee451s2019

import org.apache.spark.ml.feature.{CountVectorizer, IDF}
import org.apache.spark.sql.{Column, Dataset, SparkSession}
import org.apache.spark.sql.functions.size

object DocumentTermFrequency {

	final val TITLE_COL_NAME = "Title"
	final val TERMS_COL_NAME = "Terms"
	final val TERM_FREQ_COL_NAME = "Term-freq"
	final val TERM_IDF_COL_NAME = "Term-IDF"

	def get(sparkSession: SparkSession, documentTermMatrix: Dataset[(String, Seq[String])], numTerms: Option[Int] = None) : DocumentTermFrequencyMatrix = {
		val filtered = documentTermMatrix.toDF(TITLE_COL_NAME, TERMS_COL_NAME).where(size(new Column(TERMS_COL_NAME)) > 1)

		val countVectorizer = new CountVectorizer().setInputCol(TERMS_COL_NAME).setOutputCol(TERM_FREQ_COL_NAME)
		if (!numTerms.isEmpty) {
			countVectorizer.setVocabSize(numTerms.get)
		}
		val vocabModel = countVectorizer.fit(filtered)
		val docTermFreqMatrix = vocabModel.transform(filtered)
		docTermFreqMatrix.cache()

		val idf = new IDF().setInputCol(TERM_FREQ_COL_NAME).setOutputCol(TERM_IDF_COL_NAME)
		val idfModel = idf.fit(docTermFreqMatrix)
		val idfMatrix = idfModel.transform(docTermFreqMatrix).select(TITLE_COL_NAME, TERM_IDF_COL_NAME)

		val docIds = docTermFreqMatrix.rdd.map(_.getString(0)).zipWithUniqueId().map(_.swap).collect().toMap
		val termIds = vocabModel.vocabulary
		val idfScale = idfModel.idf.toArray

		new DocumentTermFrequencyMatrix(idfMatrix, docIds, termIds, idfScale)
	}
}
