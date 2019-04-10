
#include<iostream>

#include"../ParallelTDIDF/tdidf.cuh"
#include"../LargeMatrixParallelSVD/svd.cuh"
#include"lsa.cuh"

const auto test_TD_docs = 12;
const auto test_TD_terms = 9;

static double test_doc_term_freq[test_TD_docs * test_TD_terms] = {
	1, 0, 0, 1, 0, 0, 0, 0, 0,
	1, 0, 1, 0, 0, 0, 0, 0, 0,
	1, 1, 0, 0, 0, 0, 0, 0, 0,
	0, 1, 1, 0, 1, 0, 0, 0, 0,
	0, 1, 1, 2, 0, 0, 0, 0, 0,
	0, 1, 0, 0, 1, 0, 0, 0, 0,
	0, 1, 0, 0, 1, 0, 0, 0, 0,
	0, 0, 1, 1, 0, 0, 0, 0, 0,
	0, 1, 0, 0, 0, 0, 0, 0, 1,
	0, 0, 0, 0, 0, 1, 1, 1, 0,
	0, 0, 0, 0, 0, 0, 1, 1, 1,
	0, 0, 0, 0, 0, 0, 0, 1, 1,
};

const auto test_k = 2;

const auto numConcept = 2;
const auto numTerms = 2;
const auto numDocs = 2;

void printColMajorMatrix(int m, int n, const double*A, int lda, const char* name) {
	for (int row = 0; row < m; row++) {
		for (int col = 0; col < n; col++) {
			double Areg = A[col * m + row];
			printf("%s(%d,%d) = %f\n", name, row, col, Areg);
		}
	}
}

void printRowMajorMatrix(int m, int n, const double*A, int lda, const char* name) {
	for (int row = 0; row < m; row++) {
		for (int col = 0; col < n; col++) {
			double Areg = A[row * n + col];
			printf("%s(%d,%d) = %f\n", name, row, col, Areg);
		}
	}
}

void printConcepts(const int numConcept, const int numTerms, const int numDocs, const std::vector<std::vector<std::tuple<int, double>>> topTerms, const std::vector<std::vector<std::tuple<int, double>>> topDocs) {
	for (int c = 0; c < numConcept; c++) {
		std::cout << "Concept " << c << ":" << std::endl;
		std::cout << "\t" << "Top Term Idx: ";
		for (int i = 0; i < numTerms; i++) {
			std::cout << std::get<0>(topTerms[c][i]) << "(" << std::get<1>(topTerms[c][i]) << "), ";
		}
		std::cout << std::endl;
		std::cout << "\t" << "Top Doc Idx: ";
		for (int i = 0; i < numDocs; i++) {
			std::cout << std::get<0>(topDocs[c][i]) << "(" << std::get<1>(topDocs[c][i]) << "), ";
		}
		std::cout << std::endl;
	}
}

int main(){
	const int m = test_TD_docs;
	const int n = test_TD_terms;
	const int k = test_k;
	const int lda = m;
	//TD->IDF
	//TODO:
	//svd
	double U[lda * k];
	double S[k];
	double VT[k * n];
	printf("A = \n");
	printColMajorMatrix(m, n, test_doc_term_freq, lda, "A");
	printf("=====\n");
	approximate_svd(m, n, k, test_doc_term_freq, U, S, VT);
	printf("U = \n");
	printRowMajorMatrix(m, k, U, lda, "U");
	printf("=====\n");
	printf("S = \n");
	printRowMajorMatrix(k, 1, S, lda, "S");
	printf("=====\n");
	printf("VT = \n");
	printRowMajorMatrix(k, n, VT, lda, "VT");
	printf("=====\n");
	//concepts
	auto topTerms = topTermsInTopConcepts(k, n, VT, numConcept, numTerms);
	auto topDocs = topDocsInTopConcepts(m, k, U, numConcept, numDocs);
	printConcepts(numConcept, numTerms, numDocs, topTerms, topDocs);
	//corrolated
	//TODO:
    return 0;
}