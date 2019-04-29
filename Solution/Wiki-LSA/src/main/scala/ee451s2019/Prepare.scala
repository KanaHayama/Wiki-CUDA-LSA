package ee451s2019

import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.{Dataset, SparkSession}

object Prepare {
	private final val STOP_WORD_FILENAME = "/stopwords.txt"

	def main(args: Array[String])= {
		// args
		val filename = args(0)
		val numConcepts = if (args.length > 1) args(1).toInt else 10000
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
		timing.stop("Doc-terms")

		// TF/IDF matrix
		timing.restart()
		val documentTermFrequencyMatrix = DocumentTermFrequency.get(sparkSession, documentTermMatrix, Some(numTerms))
		timing.stop("TD-IDF")

		// SVD
		timing.restart()
		val svd = MLlibSVD.get(sparkSession, documentTermFrequencyMatrix, numConcepts)
		timing.stop("SVD")

		// print static results
		timing.restart()
		val findTops = new FindTops(svd, documentTermFrequencyMatrix.docIds, documentTermFrequencyMatrix.termIds)
		val topDocs = findTops.topDocsInTopConcepts(numShowConcepts, numShowDocs)
		val topTerms = findTops.topTermsInTopConcepts(numShowConcepts, numShowTerms)
		for (conceptId <- 0 until numShowConcepts) {
			println("Concept %d: ".format(conceptId))
			println("\tDocs: " + topDocs(conceptId).map(_._1).mkString(", "))
			println("\tTerms: " + topTerms(conceptId).map(_._1).mkString(", "))
		}
		timing.stop("Find Top")

		// save
		//TODO:

		// sample queries
		val queryEngine = new LSAQueryEngine(svd, documentTermFrequencyMatrix.docIds, documentTermFrequencyMatrix.termIds, documentTermFrequencyMatrix.idfScale)
		queryEngine.printTopTermsForTerm("algorithm")
		queryEngine.printTopTermsForTerm("radiohead")
		queryEngine.printTopTermsForTerm("tarantino")

		queryEngine.printTopDocsForTerm("fir")
		queryEngine.printTopDocsForTerm("graph")

		queryEngine.printTopDocsForDoc("Romania")
		queryEngine.printTopDocsForDoc("Brad Pitt")
		queryEngine.printTopDocsForDoc("Radiohead")

		queryEngine.printTopDocsForTermQuery(Seq("factorization", "decomposition"))

	}
}
