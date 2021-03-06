package ee451s2019

import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.{Dataset, SparkSession}

object Prepare {
	private final val STOP_WORD_FILENAME = "/stopwords.txt"

	def main(args: Array[String]): (DocumentTermFrequencyMatrix, SparkSession, Int, Int, Int, Int) = {
		// args
		val filename = args(0)
		val numConcepts = if (args.length > 1) args(1).toInt else 100
		val numTerms = if (args.length > 2) args(2).toInt else 20000

		// init
		val sparkSession = SparkSession.builder().config("spark.serializer", classOf[KryoSerializer].getName).getOrCreate() // Use KryoSerializer rather than default Java serializer for good performance
		val numShowConcepts = 10
		val numShowDocs = 10
		val numShowTerms = 10
		assert(numConcepts >= numShowConcepts)
		assert(numTerms >= numShowTerms)
		val timing = new Timing()

		// Convert XML file to doc-content mapping
		timing.restart()
		val documentContentMatrix: Dataset[(String, String)] = DocumentContent.get(sparkSession, filename)
		println("Loaded %d documents".format(documentContentMatrix.count()))
		timing.stop("Doc-content")

		// Separate content to terms
		timing.restart()
		val stopWords = scala.io.Source.fromInputStream(getClass.getResourceAsStream(STOP_WORD_FILENAME)).getLines().toSet
		val documentTermMatrix = DocumentTerm.get(sparkSession, documentContentMatrix, stopWords)
		documentContentMatrix.cache() //cache for later use
		documentContentMatrix.count() //count is a action, so do the action actually, to get exec time
		timing.stop("Doc-terms")

		// TF/IDF matrix
		timing.restart()
		val documentTermFrequencyMatrix = DocumentTermFrequency.get(sparkSession, documentTermMatrix, Some(numTerms))
		documentTermFrequencyMatrix.idfMatrix.cache() //cache for later use
		documentTermFrequencyMatrix.idfMatrix.count() //count is a action, so do the action actually, to get exec time
		timing.stop("TF-IDF")

		(documentTermFrequencyMatrix, sparkSession, numConcepts, numShowConcepts, numShowDocs, numShowTerms)
	}
}
