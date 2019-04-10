#pragma once
#ifndef _INC_SVD // include guard for 3rd party interop
#define _INC_SVD

void svd(int rows, int cols, double * matrix_A, double * matrix_U, double * array_S, double * matrix_VT);

void svd_r(int rows, int cols, double * matrix_U, double * array_S, double * matrix_VT, double * matrix_A);

void approximate_svd(int rows, int cols, int k, double * matrix_A, double * matrix_U, double * array_S, double * matrix_VT);

#endif // _INC_SVD