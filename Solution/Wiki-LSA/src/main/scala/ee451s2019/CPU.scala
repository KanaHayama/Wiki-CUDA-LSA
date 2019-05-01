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
