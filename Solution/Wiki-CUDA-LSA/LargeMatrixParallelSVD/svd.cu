#include"svd.cuh"
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

//TODO: parallelize
void row_major_2_col_major(int rows, int cols, double * matrix) {
	double * temp = new double[rows * cols];
	memcpy(temp, matrix, sizeof(double) * rows * cols);
	for (int row = 0; row < rows; row++) {
		for (int col = 0; col < cols; col++) {
			matrix[col * rows + row] = temp[row * cols + col];
		}
	}
	delete[] temp;
}

//TODO: parallelize
void col_major_2_row_major(int rows, int cols, double * matrix) {
	double * temp = new double[rows * cols];
	memcpy(temp, matrix, sizeof(double) * rows * cols);
	for (int row = 0; row < rows; row++) {
		for (int col = 0; col < cols; col++) {
			matrix[row * cols + col] = temp[col * rows + row];
		}
	}
	delete[] temp;
}

void svd(int rows, int cols, double * matrix_A, double * matrix_U, double * array_S, double * matrix_VT) {
	assert(rows >= cols);
	row_major_2_col_major(rows, cols, matrix_A);
	// step 1: create cusolverDn handle
	cusolverDnHandle_t cusolverH = NULL;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
	cusolver_status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

	// step 2: copy A and B to device
	const int lda = rows;
	cudaError_t cudaStat = cudaSuccess;
	double * d_A = NULL;
	cudaStat = cudaMalloc((void**)&d_A, sizeof(double) * lda * cols);
	assert(cudaSuccess == cudaStat);
	double *d_U = NULL;
	cudaStat = cudaMalloc((void**)&d_U, sizeof(double) * lda * rows);
	assert(cudaSuccess == cudaStat);
	double *d_S = NULL;
	cudaStat = cudaMalloc((void**)&d_S, sizeof(double) * cols);
	assert(cudaSuccess == cudaStat);
	double *d_VT = NULL;
	cudaStat = cudaMalloc((void**)&d_VT, sizeof(double) * lda * cols);
	assert(cudaSuccess == cudaStat);
	int * devInfo = NULL;
	cudaStat = cudaMalloc((void**)&devInfo, sizeof(int));
	assert(cudaSuccess == cudaStat);

	cudaStat = cudaMemcpy(d_A, matrix_A, sizeof(double) * lda * cols, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat);

	// step 3: query working space of SVD
	int lwork = 0;
	cusolver_status = cusolverDnDgesvd_bufferSize(
		cusolverH,
		rows,
		cols,
		&lwork);
	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

	double *d_work = NULL;
	cudaStat = cudaMalloc((void**)&d_work, sizeof(double) * lwork);
	assert(cudaSuccess == cudaStat);

	// step 4: compute SVD
	signed char jobu = 'A'; // all m columns of U
	signed char jobvt = 'A'; // all n columns of VT
	double *d_rwork = NULL;
	cusolver_status = cusolverDnDgesvd(
		cusolverH,
		jobu,
		jobvt,
		rows,
		cols,
		d_A,
		lda,
		d_S,
		d_U,
		lda,  // ldu
		d_VT,
		lda, // ldvt,
		d_work,
		lwork,
		d_rwork,
		devInfo);
	cudaStat = cudaDeviceSynchronize();
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
	assert(cudaSuccess == cudaStat);

	// step 5: copy back
	if (matrix_U) {
		cudaStat = cudaMemcpy(matrix_U, d_U, sizeof(double) * lda * rows, cudaMemcpyDeviceToHost);
		assert(cudaSuccess == cudaStat);
		col_major_2_row_major(lda, rows, matrix_U);
	}
	if (array_S) {
		cudaStat = cudaMemcpy(array_S, d_S, sizeof(double) * cols, cudaMemcpyDeviceToHost);
		assert(cudaSuccess == cudaStat);
	}
	if (matrix_VT) {
		cudaStat = cudaMemcpy(matrix_VT, d_VT, sizeof(double) * lda * cols, cudaMemcpyDeviceToHost);
		assert(cudaSuccess == cudaStat);
		col_major_2_row_major(lda, cols, matrix_VT);
	}
	int info_gpu = 0;
	cudaStat = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
	assert(cudaSuccess == cudaStat);
	assert(0 == info_gpu);

	// step 6: free
	if (d_A) cudaFree(d_A);
	if (d_S) cudaFree(d_S);
	if (d_U) cudaFree(d_U);
	if (d_VT) cudaFree(d_VT);
	if (devInfo) cudaFree(devInfo);
	if (d_work) cudaFree(d_work);
	if (d_rwork) cudaFree(d_rwork);
	if (cusolverH) cusolverDnDestroy(cusolverH);

	cudaDeviceReset();
}

void svd_r(int rows, int cols, double * matrix_U, double * array_S, double * matrix_VT, double * matrix_A) {
	// copy
	const int lda = rows;
	cudaError_t cudaStat = cudaSuccess;
	double *d_U = NULL;
	cudaStat = cudaMalloc((void**)&d_U, sizeof(double) * lda * rows);
	assert(cudaSuccess == cudaStat);
	double *d_S = NULL;
	cudaStat = cudaMalloc((void**)&d_S, sizeof(double) * cols);
	assert(cudaSuccess == cudaStat);
	double *d_VT = NULL;
	cudaStat = cudaMalloc((void**)&d_VT, sizeof(double) * lda * cols);
	assert(cudaSuccess == cudaStat);
	double * d_W = NULL;  // W = S*VT
	cudaStat = cudaMalloc((void**)&d_W, sizeof(double) * lda * cols);
	assert(cudaSuccess == cudaStat);
	double * d_A = NULL;
	cudaStat = cudaMalloc((void**)&d_A, sizeof(double) * lda * cols);
	assert(cudaSuccess == cudaStat);
	int * devInfo = NULL;
	cudaStat = cudaMalloc((void**)&devInfo, sizeof(int));
	assert(cudaSuccess == cudaStat);

	cudaStat = cudaMemcpy(d_U, matrix_U, sizeof(double) * lda * rows, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat);
	cudaStat = cudaMemcpy(d_S, array_S, sizeof(double) * cols, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat);
	cudaStat = cudaMemcpy(d_VT, matrix_VT, sizeof(double) * lda * cols, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat);

	// create cublas handle
	cublasHandle_t cublasH = NULL;
	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
	cublas_status = cublasCreate(&cublasH);
	assert(CUBLAS_STATUS_SUCCESS == cublas_status);

	// W = S*VT
	cublas_status = cublasDdgmm(
		cublasH,
		CUBLAS_SIDE_LEFT,
		cols,
		cols,
		d_VT,
		lda,
		d_S,
		1,
		d_W,
		lda);
	assert(CUBLAS_STATUS_SUCCESS == cublas_status);

	// A := U*W
	assert(cudaSuccess == cudaStat);
	const double h_one = 1;
	const double h_zero = 0;
	cublas_status = cublasDgemm_v2(
		cublasH,
		CUBLAS_OP_N, // U
		CUBLAS_OP_N, // W
		rows, // number of rows of A
		cols, // number of columns of A
		cols, // number of columns of U 
		&h_one, /* host pointer */
		d_U, // U
		lda,
		d_W, // W
		lda,
		&h_zero, /* hostpointer */
		d_A,
		lda);
	assert(CUBLAS_STATUS_SUCCESS == cublas_status);

	// copy
	if (matrix_A) {
		cudaStat = cudaMemcpy(matrix_A, d_A, sizeof(double) * lda * cols, cudaMemcpyDeviceToHost);
		assert(cudaSuccess == cudaStat);
	}

	// free
	if (d_A) cudaFree(d_A);
	if (d_S) cudaFree(d_S);
	if (d_U) cudaFree(d_U);
	if (d_VT) cudaFree(d_VT);
	if (devInfo) cudaFree(devInfo);
	if (cublasH) cublasDestroy(cublasH);

	cudaDeviceReset();
}

//TODO: use k to speed up svd, rather than apply after svd
void approximate_svd(int rows, int cols, int k, double * matrix_A, double * matrix_U, double * array_S, double * matrix_VT) {
	assert(rows >= cols);
	assert(k <= rows && k <= cols);
	row_major_2_col_major(rows, cols, matrix_A);
	// step 1: create cusolverDn handle
	cusolverDnHandle_t cusolverH = NULL;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
	cusolver_status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

	// step 2: copy A and B to device
	const int lda = rows;
	cudaError_t cudaStat = cudaSuccess;
	double * d_A = NULL;
	cudaStat = cudaMalloc((void**)&d_A, sizeof(double) * lda * cols);
	assert(cudaSuccess == cudaStat);
	double *d_U = NULL;
	cudaStat = cudaMalloc((void**)&d_U, sizeof(double) * lda * rows);
	assert(cudaSuccess == cudaStat);
	double *d_S = NULL;
	cudaStat = cudaMalloc((void**)&d_S, sizeof(double) * cols);
	assert(cudaSuccess == cudaStat);
	double *d_VT = NULL;
	cudaStat = cudaMalloc((void**)&d_VT, sizeof(double) * lda * cols);
	assert(cudaSuccess == cudaStat);
	int * devInfo = NULL;
	cudaStat = cudaMalloc((void**)&devInfo, sizeof(int));
	assert(cudaSuccess == cudaStat);

	cudaStat = cudaMemcpy(d_A, matrix_A, sizeof(double) * lda * cols, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat);

	// step 3: query working space of SVD
	int lwork = 0;
	cusolver_status = cusolverDnDgesvd_bufferSize(
		cusolverH,
		rows,
		cols,
		&lwork);
	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

	double *d_work = NULL;
	cudaStat = cudaMalloc((void**)&d_work, sizeof(double) * lwork);
	assert(cudaSuccess == cudaStat);

	// step 4: compute SVD
	signed char jobu = 'A'; // all m columns of U
	signed char jobvt = 'A'; // all n columns of VT
	double *d_rwork = NULL;
	cusolver_status = cusolverDnDgesvd(
		cusolverH,
		jobu,
		jobvt,
		rows,
		cols,
		d_A,
		lda,
		d_S,
		d_U,
		lda,  // ldu
		d_VT,
		lda, // ldvt,
		d_work,
		lwork,
		d_rwork,
		devInfo);
	cudaStat = cudaDeviceSynchronize();
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
	assert(cudaSuccess == cudaStat);

	// step 5: copy back
	if (matrix_U) {
		cudaStat = cudaMemcpy(matrix_U, d_U, sizeof(double) * lda * k, cudaMemcpyDeviceToHost);
		assert(cudaSuccess == cudaStat);
		col_major_2_row_major(lda, k, matrix_U);
	}
	if (array_S) {
		cudaStat = cudaMemcpy(array_S, d_S, sizeof(double) * k, cudaMemcpyDeviceToHost);
		assert(cudaSuccess == cudaStat);
	}
	if (matrix_VT) {
		for (int i = 0; i < cols; i++) {
			cudaStat = cudaMemcpy(matrix_VT + i * k, d_VT + i * lda, sizeof(double) * k, cudaMemcpyDeviceToHost);
			assert(cudaSuccess == cudaStat);
		}
		col_major_2_row_major(k, cols, matrix_VT);
	}
	int info_gpu = 0;
	cudaStat = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
	assert(cudaSuccess == cudaStat);
	assert(0 == info_gpu);

	// step 6: free
	if (d_A) cudaFree(d_A);
	if (d_S) cudaFree(d_S);
	if (d_U) cudaFree(d_U);
	if (d_VT) cudaFree(d_VT);
	if (devInfo) cudaFree(devInfo);
	if (d_work) cudaFree(d_work);
	if (d_rwork) cudaFree(d_rwork);
	if (cusolverH) cusolverDnDestroy(cusolverH);

	cudaDeviceReset();
}