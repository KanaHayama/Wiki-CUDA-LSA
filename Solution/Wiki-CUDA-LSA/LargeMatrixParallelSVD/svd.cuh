#pragma once
#ifndef _INC_SVD // include guard for 3rd party interop
#define _INC_SVD

void svd(int rows, int cols, double * matrix_A, double * matrix_U, double * array_S, double * matrix_VT);

void principle_k_singulars(int cols, int k, double * array_S);

void svd_r(int rows, int cols, double * matrix_U, double * array_S, double * matrix_VT, double * matrix_A);

//TODO: use k to speed up svd, rather than apply k after svd

#endif // _INC_SVD