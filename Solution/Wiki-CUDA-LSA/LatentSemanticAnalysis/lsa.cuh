#pragma once
#ifndef _INC_LSA // include guard for 3rd party interop
#define _INC_LSA

#include<vector>
#include<tuple>

std::vector<std::vector<std::tuple<int, float>>> topTermsInTopConcepts(int rows, int cols, float * matrix_VT, int numConcepts, int numTerms);

std::vector<std::vector<std::tuple<int, float>>> topDocsInTopConcepts(int rows, int cols, float * matrix_U, int numConcepts, int numDocs);

void rowsNormalized(int rows, int cols, float * matrix);

std::vector<std::tuple<int, float>> topTermsForTerm(int rows, int cols, float * normVS, int termIdx);

#endif // _INC_LSA