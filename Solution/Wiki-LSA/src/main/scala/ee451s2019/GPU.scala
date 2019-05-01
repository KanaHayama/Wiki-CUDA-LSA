package ee451s2019

object GPU {
	def main(args: Array[String])= {
		val (documentTermFrequencyMatrix, sparkSession, numConcepts, numShowConcepts, numShowDocs, numShowTerms) = Prepare.main(args)

		val timing = new Timing()
		// Mahout SVD
		timing.restart()
		val _ = MahoutSVD.get(sparkSession, documentTermFrequencyMatrix, numConcepts)
		timing.stop("Mahout SVD")

	}
}
