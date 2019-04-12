#include"lsa.cuh"
#include<svd.cuh>

#include<cassert>
#include<algorithm>

//TODO: return void
//TODO: parallelize
std::vector<std::vector<std::tuple<int, float>>> topElementsInTopConcepts(int rows, int cols, float * matrix, int numConcepts, int numElement) {
	assert(numConcepts <= cols && numElement <= rows);
	auto topElements = std::vector<std::vector<std::tuple<int, float>>>();
	//U: doc->row, concept->col
	for (int j = 0; j < numConcepts; j++) {
		auto elementWeights = std::vector<std::tuple<int, float>>();
		for (int i = 0; i < rows; i++) {
			elementWeights.push_back(std::make_tuple(i, matrix[i * cols + j]));
		}
		sort(elementWeights.begin(), elementWeights.end(), [](std::tuple<int, float> a, std::tuple<int, float> b) {return std::get<1>(a) > std::get<1>(b); });//weight desc
		auto conceptTopElements = std::vector<std::tuple<int, float>>();
		for (int i = 0; i < numElement; i++) {
			conceptTopElements.push_back(elementWeights[i]);
		}
		topElements.push_back(conceptTopElements);
	}
	return topElements;
}


//TODO: parallelize
void rowsNormalized(int rows, int cols, float * matrix) {
	for (int i = 0; i < rows; i++) {
		float sqrSum = 0;
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
std::vector<std::tuple<int, float>> topsForTerm(int rows, int cols, float * normXS, int vRows, int vCols, float * normVS, int termIdx) {
	assert(cols == vCols);
	float * rowVec = new float[vCols];
	memcpy(rowVec, normVS + termIdx * vCols, sizeof(float) * vCols);
	float * resultVec = new float[rows];
	multiply(rows, cols, 1, normXS, rowVec, resultVec);
	auto docScores = std::vector<std::tuple<int, float>>();
	for (int i = 0; i < rows; i++) {
		docScores.push_back(std::make_tuple(i, resultVec[i]));
	}
	sort(docScores.begin(), docScores.end(), [](std::tuple<int, float> a, std::tuple<int, float> b) {return std::get<1>(a) > std::get<1>(b); });//weight desc
	delete[] resultVec;
	delete[] rowVec;
	return docScores;
}