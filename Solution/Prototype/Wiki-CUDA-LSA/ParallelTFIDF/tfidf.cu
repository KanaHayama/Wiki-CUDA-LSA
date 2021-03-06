#include"tfidf.cuh"

#include<cmath>

//TODO: parallelize
void tfidf(int rows, int cols, float * matrix) {
	int numDocs = rows;
	int numTerms = cols;
	for (int termIdx = 0; termIdx < numTerms; termIdx++) {
		int includeDocs = 0;
		for (int docIdx = 0; docIdx < numDocs; docIdx++) {
			if (matrix[docIdx * cols + termIdx] > 0) {
				includeDocs++;
			}
		}
		float termIdf = log10(numDocs / float(includeDocs + 1));
		for (int docIdx = 0; docIdx < numDocs; docIdx++) {
			matrix[docIdx * cols + termIdx] *= termIdf;
		}
	}
}