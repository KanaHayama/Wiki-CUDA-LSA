
#include<iostream>
#include<map>
#include<tfidf.cuh>
#include<svd.cuh>
#include"lsa.cuh"

const auto test_TD_docs = 12;
const auto test_TD_terms = 9;

static float test_doc_term_freq[test_TD_docs * test_TD_terms] = {
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

void printColMajorMatrix(int m, int n, const float*A, int lda, const char* name) {
	for (int row = 0; row < m; row++) {
		for (int col = 0; col < n; col++) {
			float Areg = A[col * m + row];
			printf("%s(%d,%d) = %f\n", name, row, col, Areg);
		}
	}
}

void printRowMajorMatrix(int m, int n, const float*A, const char* name) {
	for (int row = 0; row < m; row++) {
		for (int col = 0; col < n; col++) {
			float Areg = A[row * n + col];
			printf("%s(%d,%d) = %f\n", name, row, col, Areg);
		}
	}
}

void printConcepts(const int numConcept, const int numTerms, const int numDocs, const std::vector<std::vector<std::tuple<int, float>>> topTerms, const std::vector<std::vector<std::tuple<int, float>>> topDocs) {
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

void printTermTerm(const int lookUpIdx, const std::vector<std::tuple<int, float>> termTerm) {
	std::cout << "Term correlation to term index " << lookUpIdx << " ranking:" << std::endl;
	std::cout << "\t";
	for (auto & t : termTerm) {
		std::cout << std::get<0>(t) << "(" << std::get<1>(t) << "), ";
	}
	std::cout << std::endl;
}

void routine() {
	const int m = test_TD_docs;
	const int n = test_TD_terms;
	const int k = test_k;
	const int lda = m;
	//TF/IDF
	tfidf(m, n, test_doc_term_freq);
	//svd
	float U[lda * k];
	float S[k];
	float VT[k * n];
	printf("A = \n");
	printColMajorMatrix(m, n, test_doc_term_freq, lda, "A");
	printf("=====\n");
	approximate_svd(m, n, k, test_doc_term_freq, U, S, VT);
	printf("U = \n");
	printRowMajorMatrix(m, k, U, "U");
	printf("=====\n");
	printf("S = \n");
	printRowMajorMatrix(k, 1, S, "S");
	printf("=====\n");
	printf("VT = \n");
	printRowMajorMatrix(k, n, VT, "VT");
	printf("=====\n");
	printf("V = \n");
	float V[n * k];
	memcpy(V, VT, sizeof(float) * k * n);
	transpose(k, n, V);
	printRowMajorMatrix(n, k, V, "V");
	printf("=====\n");
	//concepts
	auto topTerms = topElementsInTopConcepts(n, k, V, numConcept, numTerms);
	auto topDocs = topElementsInTopConcepts(m, k, U, numConcept, numDocs);
	printConcepts(numConcept, numTerms, numDocs, topTerms, topDocs);
	printf("=====\n");
	//corrolated
	printf("V*S = \n");
	float VS[n * k];
	memcpy(VS, V, sizeof(float) * k * n);
	multiplyByDiagonalMatrix(n, k, VS, S);
	printRowMajorMatrix(n, k, VS, "VS");
	printf("=====\n");

	printf("norm(V*S) = \n");
	float normVS[n * k];
	memcpy(normVS, VS, sizeof(float) * k * n);
	rowsNormalized(n, k, normVS);
	printRowMajorMatrix(n, k, normVS, "normVS");
	printf("=====\n");

	printf("U*S = \n");
	float US[m * k];
	memcpy(US, U, sizeof(float) * k * m);
	multiplyByDiagonalMatrix(m, k, US, S);
	printRowMajorMatrix(m, k, US, "US");
	printf("=====\n");

	printf("norm(U*S) = \n");
	float normUS[m * k];
	memcpy(normUS, US, sizeof(float) * k * m);
	rowsNormalized(m, k, normUS);
	printRowMajorMatrix(m, k, normUS, "normUS");
	printf("=====\n");

	for (int lookUpTermIdx = 0; lookUpTermIdx < 3; lookUpTermIdx++) {
		auto termTerm = topsForTerm(n, k, normVS, n, k, normVS, lookUpTermIdx);
		printTermTerm(lookUpTermIdx, termTerm);
	}
}
/*
int main() {
	routine();
	return 0;
}
*/