/*
 * How to compile (assume cuda is installed at /usr/local/cuda/)
 *   nvcc -c -I/usr/local/cuda/include svd_example.cpp
 *   g++ -o a.out svd_example.o -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusolver
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "svd.cuh"

void printMatrix(int m, int n, const double*A, int lda, const char* name) {
	for (int row = 0; row < m; row++) {
		for (int col = 0; col < n; col++) {
			double Areg = A[row * n + col];
			printf("%s(%d,%d) = %f\n", name, row, col, Areg);
		}
	}
}

void test_precise() {
	const int m = 3;
	const int n = 2;
	const int lda = m;
	/*       | 1 4  |
	 *   A = | 2 2  |
	 *       | 5 1  |
	 */
	double A[lda*n] = { 1.0, 4.0, 2.0, 2.0, 5.0, 1.0 };
	double result_A[lda*n];
	double U[lda*m]; // m-by-m unitary matrix 
	double VT[lda*n];  // n-by-n unitary matrix
	double S[n]; // singular value

	printf("A = (matlab base-1)\n");
	printMatrix(m, n, A, lda, "A");
	printf("=====\n");

	svd(m, n, A, U, S, VT);

	printf("=====\n");

	printf("S = (matlab base-1)\n");
	printMatrix(n, 1, S, lda, "S");
	printf("=====\n");

	printf("U = (matlab base-1)\n");
	printMatrix(m, m, U, lda, "U");
	printf("=====\n");

	printf("VT = (matlab base-1)\n");
	printMatrix(n, n, VT, lda, "VT");
	printf("=====\n");

	svd_r(m, n, U, S, VT, result_A);
	printMatrix(m, n, result_A, lda, "rsult A");
}

void test_approximate() {
	const int m = 3;
	const int n = 2;
	const int lda = m;
	/*       | 1 4  |
	 *   A = | 2 2  |
	 *       | 5 1  |
	 */
	const int k = 2;
	double A[lda*n] = { 1.0, 4.0, 2.0, 2.0, 5.0, 1.0 };
	double result_A[lda*n];
	double U[lda * k]; // m-by-m unitary matrix 
	double S[k]; // singular value
	double VT[k * n];  // n-by-n unitary matrix	

	printf("A = (matlab base-1)\n");
	printMatrix(m, n, A, lda, "A");
	printf("=====\n");

	aproximate_svd(m, n, k, A, U, S, VT);

	printf("=====\n");

	printf("S = (matlab base-1)\n");
	printMatrix(k, 1, S, lda, "S");
	printf("=====\n");

	printf("U = (matlab base-1)\n");
	printMatrix(m, k, U, lda, "U");
	printf("=====\n");

	printf("VT = (matlab base-1)\n");
	printMatrix(k, n, VT, lda, "VT");
	printf("=====\n");
}

int main(int argc, char*argv[]) {
	test_precise();
	test_approximate();

	return 0;
}