#pragma once
#ifndef _INC_SVD // include guard for 3rd party interop
#define _INC_SVD

void transpose(int rows, int cols, float * matrix);

void svd(int rows, int cols, float * matrix_A, float * matrix_U, float * array_S, float * matrix_VT);

void svd_r(int rows, int cols, float * matrix_U, float * array_S, float * matrix_VT, float * matrix_A);

void approximate_svd(int rows, int cols, int k, float * matrix_A, float * matrix_U, float * array_S, float * matrix_VT);

void multiplyByDiagonalMatrix(int rows, int cols, float * matrix, float * array);

void multiply(int rows, int k, int cols, float * matrixA, float * matrixB, float * result);

#endif // _INC_SVD