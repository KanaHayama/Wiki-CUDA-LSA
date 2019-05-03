package ee451s2019

import breeze.linalg.{DenseMatrix => BDenseMatrix, SparseVector => BSparseVector}

import org.apache.spark.mllib.linalg.{Vector => MLLibVector}
import org.apache.spark.mllib.linalg.{Matrices, Matrix, SingularValueDecomposition, Vectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix

import scala.collection.Map

/**
  * LSA Query Engine. This class is based on the code of Book "Advanced Analytics with Spark, 2nd Edition" by Sandy Ryza, Uri Laserson, Sean Owen, and Josh Wills
  * @param svd
  * @param docIds
  * @param termIds
  * @param termIdfs
  */
class LSAQueryEngine(val svd: SingularValueDecomposition[RowMatrix, Matrix], val docIds: Map[Long, String], val termIds: Array[String], val termIdfs: Array[Double]) {

	val VS: BDenseMatrix[Double] = multiplyByDiagonalMatrix(svd.V, svd.s)
	val normalizedVS: BDenseMatrix[Double] = rowsNormalized(VS)
	val US: RowMatrix = multiplyByDiagonalRowMatrix(svd.U, svd.s)
	val normalizedUS: RowMatrix = distributedRowsNormalized(US)

	val idTerms: Map[String, Int] = termIds.zipWithIndex.toMap
	val idDocs: Map[String, Long] = docIds.map(_.swap)

	/**
	  * Finds the product of a dense matrix and a diagonal matrix represented by a vector.
	  * Breeze doesn't support efficient diagonal representations, so multiply manually.
	  */
	private def multiplyByDiagonalMatrix(mat: Matrix, diag: MLLibVector): BDenseMatrix[Double] = {
		val sArr = diag.toArray
		new BDenseMatrix[Double](mat.numRows, mat.numCols, mat.toArray)
			.mapPairs { case ((row, col), v) => v * sArr(col) }
	}

	/**
	  * Finds the product of a distributed matrix and a diagonal matrix represented by a vector.
	  */
	private def multiplyByDiagonalRowMatrix(mat: RowMatrix, diag: MLLibVector): RowMatrix = {
		val sArr = diag.toArray
		new RowMatrix(mat.rows.map { vec =>
			val vecArr = vec.toArray
			val newArr = (0 until vec.size).toArray.map(i => vecArr(i) * sArr(i))
			Vectors.dense(newArr)
		})
	}

	/**
	  * Returns a matrix where each row is divided by its length.
	  */
	private def rowsNormalized(mat: BDenseMatrix[Double]): BDenseMatrix[Double] = {
		val newMat = new BDenseMatrix[Double](mat.rows, mat.cols)
		for (r <- 0 until mat.rows) {
			val length = math.sqrt((0 until mat.cols).map(c => mat(r, c) * mat(r, c)).sum)
			(0 until mat.cols).foreach(c => newMat.update(r, c, mat(r, c) / length))
		}
		newMat
	}

	/**
	  * Returns a distributed matrix where each row is divided by its length.
	  */
	private def distributedRowsNormalized(mat: RowMatrix): RowMatrix = {
		new RowMatrix(mat.rows.map { vec =>
			val array = vec.toArray
			val length = math.sqrt(array.map(x => x * x).sum)
			Vectors.dense(array.map(_ / length))
		})
	}

	/**
	  * Finds docs relevant to a term. Returns the doc IDs and scores for the docs with the highest
	  * relevance scores to the given term.
	  */
	def topDocsForTerm(termId: Int): Seq[(Double, Long)] = {
		val rowArr = (0 until svd.V.numCols).map(i => svd.V(termId, i)).toArray
		val rowVec = Matrices.dense(rowArr.length, 1, rowArr)

		// Compute scores against every doc
		val docScores = US.multiply(rowVec)

		// Find the docs with the highest scores
		val allDocWeights = docScores.rows.map(_.toArray(0)).zipWithUniqueId
		allDocWeights.top(10)
	}

	/**
	  * Finds terms relevant to a term. Returns the term IDs and scores for the terms with the highest
	  * relevance scores to the given term.
	  */
	def topTermsForTerm(termId: Int): Seq[(Double, Int)] = {
		// Look up the row in VS corresponding to the given term ID.
		val rowVec = normalizedVS(termId, ::).t

		// Compute scores against every term
		val termScores = (normalizedVS * rowVec).toArray.zipWithIndex

		// Find the terms with the highest scores
		termScores.sortBy(-_._1).take(10)
	}

	/**
	  * Finds docs relevant to a doc. Returns the doc IDs and scores for the docs with the highest
	  * relevance scores to the given doc.
	  */
	def topDocsForDoc(docId: Long): Seq[(Double, Long)] = {
		// Look up the row in US corresponding to the given doc ID.
		val docRowArr = normalizedUS.rows.zipWithUniqueId.map(_.swap).lookup(docId).head.toArray
		val docRowVec = Matrices.dense(docRowArr.length, 1, docRowArr)

		// Compute scores against every doc
		val docScores = normalizedUS.multiply(docRowVec)

		// Find the docs with the highest scores
		val allDocWeights = docScores.rows.map(_.toArray(0)).zipWithUniqueId

		// Docs can end up with NaN score if their row in U is all zeros.  Filter these out.
		allDocWeights.filter(!_._1.isNaN).top(10)
	}

	/**
	  * Builds a term query vector from a set of terms.
	  */
	def termsToQueryVector(terms: Seq[String]): BSparseVector[Double] = {
		val indices = terms.map(idTerms(_)).toArray
		val values = indices.map(termIdfs(_))
		new BSparseVector[Double](indices, values, idTerms.size)
	}

	/**
	  * Finds docs relevant to a term query, represented as a vector with non-zero weights for the
	  * terms in the query.
	  */
	def topDocsForTermQuery(query: BSparseVector[Double]): Seq[(Double, Long)] = {
		val breezeV = new BDenseMatrix[Double](svd.V.numRows, svd.V.numCols, svd.V.toArray)
		val termRowArr = (breezeV.t * query).toArray

		val termRowVec = Matrices.dense(termRowArr.length, 1, termRowArr)

		// Compute scores against every doc
		val docScores = US.multiply(termRowVec)

		// Find the docs with the highest scores
		val allDocWeights = docScores.rows.map(_.toArray(0)).zipWithUniqueId
		allDocWeights.top(10)
	}

	def printTopTermsForTerm(term: String): Unit = {
		if (idTerms.contains(term)) {
			print("Top terms for term \"%s\"".format(term))
			val idWeights = topTermsForTerm(idTerms(term))
			println(idWeights.map { case (weight, id) => "\"%s\"(%s)".format(termIds(id), weight) }.mkString(", "))
		} else {
			System.err.println("Term \"%s\" not exists".format(term))
		}
	}

	def printTopDocsForDoc(doc: String): Unit = {
		if (idDocs.contains(doc)) {
			print("Top documents for document \"%s\"".format(doc))
			val idWeights = topDocsForDoc(idDocs(doc))
			println(idWeights.map { case (weight, id) => "\"%s\"(%s)".format(docIds(id), weight) }.mkString(", "))
		} else {
			System.err.println("Document \"%s\" not exists.".format(doc))
		}
	}

	def printTopDocsForTerm(term: String): Unit = {
		if (idTerms.contains(term)) {
			print("Top documents for term \"%s\"".format(term))
			val idWeights = topDocsForTerm(idTerms(term))
			println(idWeights.map { case (weight, id) => "\"%s\"(%s)".format(docIds(id), weight) }.mkString(", "))
		} else {
			System.err.println("Term \"%s\" not exists.".format(term))
		}
	}

	def printTopDocsForTermQuery(terms: Seq[String]): Unit = {
		if (terms.forall(idTerms.contains(_))) {
			print("Top documents for terms \"%s\"".format(terms.mkString(", ")))
			val queryVec = termsToQueryVector(terms)
			val idWeights = topDocsForTermQuery(queryVec)
			println(idWeights.map { case (weight, id) => "\"%s\"(%s)".format(docIds(id), weight) }.mkString(", "))
		} else {
			System.err.println("Term(s) \"%s\" not exists.".format(terms.filter(!idTerms.contains(_)).mkString(", ")))
		}

	}
}
