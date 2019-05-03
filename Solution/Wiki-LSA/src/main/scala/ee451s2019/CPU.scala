package ee451s2019

object CPU {
	def main(args: Array[String])= {
		val (documentTermFrequencyMatrix, sparkSession, numConcepts, numShowConcepts, numShowDocs, numShowTerms) = Prepare.main(args)

		val timing = new Timing()
		// MLlib SVD
		timing.restart()
		val svd = MLlibSVD.get(sparkSession, documentTermFrequencyMatrix, numConcepts)
		timing.stop("MLlib SVD")

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

		// save matrices
		//TODO:

		// sample queries
		val engine = new LSAQueryEngine(svd, documentTermFrequencyMatrix.docIds, documentTermFrequencyMatrix.termIds, documentTermFrequencyMatrix.idfScale)
		engine.printTopTermsForTerm("algorithm")
		engine.printTopTermsForTerm("radiohead")
		engine.printTopTermsForTerm("tarantino")

		engine.printTopDocsForTerm("fir")
		engine.printTopDocsForTerm("graph")

		engine.printTopDocsForDoc("Romania")
		engine.printTopDocsForDoc("Brad Pitt")
		engine.printTopDocsForDoc("Radiohead")

		engine.printTopDocsForTermQuery(Seq("factorization", "decomposition"))

		// Mahout SVD
		println("Try Mahout")
		timing.restart()
		val _ = MahoutSVD.get(sparkSession, documentTermFrequencyMatrix, numConcepts)
		timing.stop("Mahout SVD")
	}
}
