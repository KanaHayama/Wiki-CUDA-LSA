#include"lsa.cuh"
#include<svd.cuh>

#include<cassert>
#include<algorithm>


//TODO: return void
//TODO: parallelize
std::vector<std::vector<std::tuple<int, double>>> topTermsInTopConcepts(int rows, int cols, double * matrix_VT, int numConcepts, int numTerms) {
	assert(numConcepts <= rows && numTerms <= cols);
	auto topTerms = std::vector<std::vector<std::tuple<int, double>>>();
	// VT: concept->row, term->col
	for (int i = 0; i < numConcepts; i++) {
		auto termWeights = std::vector<std::tuple<int, double>>();
		for (int j = 0; j < cols; j++) {
			termWeights.push_back(std::make_tuple(j, matrix_VT[i * cols + j]));//<termIdx, termWeight>
		}
		sort(termWeights.begin(), termWeights.end(), [](auto a, auto b) {return std::get<1>(a) > std::get<1>(b); });//weight desc
		auto conceptTopTerms = std::vector<std::tuple<int, double>>();
		for (int j = 0; j < numTerms; j++) {
			conceptTopTerms.push_back(termWeights[j]);
		}
		topTerms.push_back(conceptTopTerms);
	}
	return topTerms;
}

//TODO: return void
//TODO: parallelize
std::vector<std::vector<std::tuple<int, double>>> topDocsInTopConcepts(int rows, int cols, double * matrix_U, int numConcepts, int numDocs) {
	assert(numConcepts <= cols && numDocs <= rows);
	auto topDocs = std::vector<std::vector<std::tuple<int, double>>>();
	//U: doc->row, concept->col
	for (int j = 0; j < numConcepts; j++) {
		auto docWeights = std::vector<std::tuple<int, double>>();
		for (int i = 0; i < rows; i++) {
			docWeights.push_back(std::make_tuple(i, matrix_U[i * cols + j]));
		}
		sort(docWeights.begin(), docWeights.end(), [](auto a, auto b) {return std::get<1>(a) > std::get<1>(b); });//weight desc
		auto conceptTopDocs = std::vector<std::tuple<int, double>>();
		for (int i = 0; i < numDocs; i++) {
			conceptTopDocs.push_back(docWeights[i]);
		}
		topDocs.push_back(conceptTopDocs);
	}
	return topDocs;
}


//TODO: parallelize
void rowsNormalized(int rows, int cols, double * matrix) {
	for (int i = 0; i < rows; i++) {
		double sqrSum = 0;
		for (int j = 0; j < cols; j++) {
			sqrSum += pow(matrix[i * cols + j], 2);
		}
		for (int j = 0; j < cols; j++) {
			matrix[i * cols + j] /= sqrSum;
		}
	}
}

//TODO: return void
//TODO: parallelize
std::vector<std::tuple<int, double>> topTermsForTerm(int rows, int cols, double * normVS, int termIdx) {
	double * rowVec = new double[cols];
	memcpy(rowVec, normVS + termIdx * cols, sizeof(double) * cols);
	double * resultVec = new double[rows];
	multiply(rows, cols, 1, normVS, rowVec, resultVec);
	auto termScores = std::vector<std::tuple<int, double>>();
	for (int i = 0; i < rows; i++) {
		termScores.push_back(std::make_tuple(i, resultVec[i]));
	}
	sort(termScores.begin(), termScores.end(), [](auto a, auto b) {return std::get<1>(a) > std::get<1>(b); });//weight desc
	delete[] resultVec;
	delete[] rowVec;
	return termScores;
}