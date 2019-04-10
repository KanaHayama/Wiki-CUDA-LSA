#pragma once
#ifndef _INC_SVD // include guard for 3rd party interop
#define _INC_SVD

void transpose(int rows, int cols, double * matrix);

void svd(int rows, int cols, double * matrix_A, double * matrix_U, double * array_S, double * matrix_VT);

void svd_r(int rows, int cols, double * matrix_U, double * array_S, double * matrix_VT, double * matrix_A);

void approximate_svd(int rows, int cols, int k, double * matrix_A, double * matrix_U, double * array_S, double * matrix_VT);

void multiplyByDiagonalMatrix(int rows, int cols, double * matrix, double * array);

void multiply(int rows, int k, int cols, double * matrixA, double * matrixB, double * result);

#endif // _INC_SVD