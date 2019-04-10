#pragma once
#ifndef _INC_LSA // include guard for 3rd party interop
#define _INC_LSA

#include<vector>
#include<tuple>

std::vector<std::vector<std::tuple<int, double>>> topTermsInTopConcepts(int rows, int cols, double * matrix_VT, int numConcepts, int numTerms);

std::vector<std::vector<std::tuple<int, double>>> topDocsInTopConcepts(int rows, int cols, double * matrix_U, int numConcepts, int numDocs);

void rowsNormalized(int rows, int cols, double * matrix);

std::vector<std::tuple<int, double>> topTermsForTerm(int rows, int cols, double * normVS, int termIdx);

#endif // _INC_LSA