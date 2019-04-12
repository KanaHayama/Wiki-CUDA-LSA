#pragma once
#ifndef _INC_LSA // include guard for 3rd party interop
#define _INC_LSA

#include<vector>
#include<tuple>

std::vector<std::vector<std::tuple<int, float>>> topElementsInTopConcepts(int rows, int cols, float * matrix_U, int numConcepts, int numDocs);

void rowsNormalized(int rows, int cols, float * matrix);

std::vector<std::tuple<int, float>> topsForTerm(int rows, int cols, float * normXS, int vRows, int vCols, float * normVS, int termIdx);

#endif // _INC_LSA