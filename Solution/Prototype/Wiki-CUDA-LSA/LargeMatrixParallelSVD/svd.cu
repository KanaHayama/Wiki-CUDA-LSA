#include"svd.cuh"
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

void transpose_device(int rows, int cols, float * d_Matrix) {
	if (rows == 1 || cols == 1) {
		return;
	}
	cudaError_t cudaStat = cudaSuccess;
	float * d_Result = NULL;
	cudaStat = cudaMalloc((void**)&d_Result, sizeof(float) * rows * cols);
	assert(cudaSuccess == cudaStat);
	const float alpha = 1;
	const float beta = 0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
	cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, cols, &alpha, d_Matrix, cols, &beta, d_Matrix, rows, d_Result, rows);
	assert(CUBLAS_STATUS_SUCCESS == cublas_status);
	cudaStat = cudaMemcpy(d_Matrix, d_Result, sizeof(float) * rows * cols, cudaMemcpyDeviceToDevice);
	assert(cudaSuccess == cudaStat);
	if (d_Result) cudaFree(d_Result);
	cublasDestroy(handle);
}

void transpose(int rows, int cols, float * matrix) {
	if (rows == 1 || cols == 1) {
		return;
	}
	cudaError_t cudaStat = cudaSuccess;
	float * d_Matrix = NULL;
	cudaStat = cudaMalloc((void**)&d_Matrix, sizeof(float) * rows * cols);
	assert(cudaSuccess == cudaStat);
	cudaStat = cudaMemcpy(d_Matrix, matrix, sizeof(float) * rows * cols, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat);
	transpose_device(rows, cols, d_Matrix);
	cudaStat = cudaMemcpy(matrix, d_Matrix, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost);
	assert(cudaSuccess == cudaStat);
	if (d_Matrix) cudaFree(d_Matrix);
}

inline void row_major_2_col_major_device(int rows, int cols, float * matrix) {
	transpose_device(rows, cols, matrix);
}

inline void col_major_2_row_major_device(int rows, int cols, float * matrix) {
	transpose_device(cols, rows, matrix);
}

void svd(int rows, int cols, float * matrix_A, float * matrix_U, float * array_S, float * matrix_VT) {
	assert(rows >= cols);
	// step 1: create cusolverDn handle
	cusolverDnHandle_t cusolverH = NULL;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
	cusolver_status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

	// step 2: copy A and B to device
	const int lda = rows;
	cudaError_t cudaStat = cudaSuccess;
	float * d_A = NULL;
	cudaStat = cudaMalloc((void**)&d_A, sizeof(float) * lda * cols);
	assert(cudaSuccess == cudaStat);
	float *d_U = NULL;
	cudaStat = cudaMalloc((void**)&d_U, sizeof(float) * lda * rows);
	assert(cudaSuccess == cudaStat);
	float *d_S = NULL;
	cudaStat = cudaMalloc((void**)&d_S, sizeof(float) * cols);
	assert(cudaSuccess == cudaStat);
	float *d_VT = NULL;
	cudaStat = cudaMalloc((void**)&d_VT, sizeof(float) * lda * cols);
	assert(cudaSuccess == cudaStat);
	int * devInfo = NULL;
	cudaStat = cudaMalloc((void**)&devInfo, sizeof(int));
	assert(cudaSuccess == cudaStat);

	cudaStat = cudaMemcpy(d_A, matrix_A, sizeof(float) * lda * cols, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat);

	row_major_2_col_major_device(lda, cols, d_A);

	// step 3: query working space of SVD
	int lwork = 0;
	cusolver_status = cusolverDnDgesvd_bufferSize(
		cusolverH,
		rows,
		cols,
		&lwork);
	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

	float *d_work = NULL;
	cudaStat = cudaMalloc((void**)&d_work, sizeof(float) * lwork);
	assert(cudaSuccess == cudaStat);

	// step 4: compute SVD
	signed char jobu = 'A'; // all m columns of U
	signed char jobvt = 'A'; // all n columns of VT
	float *d_rwork = NULL;
	cusolver_status = cusolverDnSgesvd(
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
		col_major_2_row_major_device(lda, rows, d_U);
		cudaStat = cudaMemcpy(matrix_U, d_U, sizeof(float) * lda * rows, cudaMemcpyDeviceToHost);
		assert(cudaSuccess == cudaStat);
	}
	if (array_S) {
		cudaStat = cudaMemcpy(array_S, d_S, sizeof(float) * cols, cudaMemcpyDeviceToHost);
		assert(cudaSuccess == cudaStat);
	}
	if (matrix_VT) {
		col_major_2_row_major_device(lda, cols, d_VT);
		cudaStat = cudaMemcpy(matrix_VT, d_VT, sizeof(float) * lda * cols, cudaMemcpyDeviceToHost);
		assert(cudaSuccess == cudaStat);
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

void svd_r(int rows, int cols, float * matrix_U, float * array_S, float * matrix_VT, float * matrix_A) {
	// copy
	const int lda = rows;
	cudaError_t cudaStat = cudaSuccess;
	float *d_U = NULL;
	cudaStat = cudaMalloc((void**)&d_U, sizeof(float) * lda * rows);
	assert(cudaSuccess == cudaStat);
	float *d_S = NULL;
	cudaStat = cudaMalloc((void**)&d_S, sizeof(float) * cols);
	assert(cudaSuccess == cudaStat);
	float *d_VT = NULL;
	cudaStat = cudaMalloc((void**)&d_VT, sizeof(float) * lda * cols);
	assert(cudaSuccess == cudaStat);
	float * d_W = NULL;  // W = S*VT
	cudaStat = cudaMalloc((void**)&d_W, sizeof(float) * lda * cols);
	assert(cudaSuccess == cudaStat);
	float * d_A = NULL;
	cudaStat = cudaMalloc((void**)&d_A, sizeof(float) * lda * cols);
	assert(cudaSuccess == cudaStat);
	int * devInfo = NULL;
	cudaStat = cudaMalloc((void**)&devInfo, sizeof(int));
	assert(cudaSuccess == cudaStat);

	cudaStat = cudaMemcpy(d_U, matrix_U, sizeof(float) * lda * rows, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat);
	cudaStat = cudaMemcpy(d_S, array_S, sizeof(float) * cols, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat);
	cudaStat = cudaMemcpy(d_VT, matrix_VT, sizeof(float) * lda * cols, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat);

	// create cublas handle
	cublasHandle_t cublasH = NULL;
	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
	cublas_status = cublasCreate(&cublasH);
	assert(CUBLAS_STATUS_SUCCESS == cublas_status);

	// W = S*VT
	cublas_status = cublasSdgmm(
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
	const float h_one = 1;
	const float h_zero = 0;
	cublas_status = cublasSgemm_v2(
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
		cudaStat = cudaMemcpy(matrix_A, d_A, sizeof(float) * lda * cols, cudaMemcpyDeviceToHost);
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
void approximate_svd(int rows, int cols, int k, float * matrix_A, float * matrix_U, float * array_S, float * matrix_VT) {
	assert(rows >= cols);
	assert(k <= rows && k <= cols);
	// step 1: create cusolverDn handle
	cusolverDnHandle_t cusolverH = NULL;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
	cusolver_status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

	// step 2: copy A and B to device
	const int lda = rows;
	cudaError_t cudaStat = cudaSuccess;
	float * d_A = NULL;
	cudaStat = cudaMalloc((void**)&d_A, sizeof(float) * lda * cols);
	assert(cudaSuccess == cudaStat);
	float *d_U = NULL;
	cudaStat = cudaMalloc((void**)&d_U, sizeof(float) * lda * rows);
	assert(cudaSuccess == cudaStat);
	float *d_S = NULL;
	cudaStat = cudaMalloc((void**)&d_S, sizeof(float) * cols);
	assert(cudaSuccess == cudaStat);
	float *d_VT = NULL;
	cudaStat = cudaMalloc((void**)&d_VT, sizeof(float) * lda * cols);
	assert(cudaSuccess == cudaStat);
	int * devInfo = NULL;
	cudaStat = cudaMalloc((void**)&devInfo, sizeof(int));
	assert(cudaSuccess == cudaStat);

	cudaStat = cudaMemcpy(d_A, matrix_A, sizeof(float) * lda * cols, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat);

	row_major_2_col_major_device(lda, cols, d_A);

	// step 3: query working space of SVD
	int lwork = 0;
	cusolver_status = cusolverDnDgesvd_bufferSize(
		cusolverH,
		rows,
		cols,
		&lwork);
	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

	float *d_work = NULL;
	cudaStat = cudaMalloc((void**)&d_work, sizeof(float) * lwork);
	assert(cudaSuccess == cudaStat);

	// step 4: compute SVD
	signed char jobu = 'A'; // all m columns of U
	signed char jobvt = 'A'; // all n columns of VT
	float *d_rwork = NULL;
	cusolver_status = cusolverDnSgesvd(
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
		col_major_2_row_major_device(lda, k, d_U);
		cudaStat = cudaMemcpy(matrix_U, d_U, sizeof(float) * lda * k, cudaMemcpyDeviceToHost);
		assert(cudaSuccess == cudaStat);
	}
	if (array_S) {
		cudaStat = cudaMemcpy(array_S, d_S, sizeof(float) * k, cudaMemcpyDeviceToHost);
		assert(cudaSuccess == cudaStat);
	}
	if (matrix_VT) {
		float * d_Temp = NULL;
		cudaStat = cudaMalloc((void**)&d_Temp, sizeof(float) * k * cols);
		assert(cudaSuccess == cudaStat);
		for (int i = 0; i < cols; i++) {
			cudaStat = cudaMemcpy(d_Temp + i * k, d_VT + i * lda, sizeof(float) * k, cudaMemcpyDeviceToDevice);
			assert(cudaSuccess == cudaStat);
		}
		col_major_2_row_major_device(k, cols, d_Temp);
		cudaStat = cudaMemcpy(matrix_VT, d_Temp, sizeof(float) * k * cols, cudaMemcpyDeviceToHost);
		assert(cudaSuccess == cudaStat);
		if (d_Temp) cudaFree(d_Temp);
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

void multiplyByDiagonalMatrix(int rows, int cols, float * matrix, float * array) {
	// copy
	cudaError_t cudaStat = cudaSuccess;
	float *d_Matrix = NULL;
	cudaStat = cudaMalloc((void**)&d_Matrix, sizeof(float) * rows * cols);
	assert(cudaSuccess == cudaStat);
	float *d_Array = NULL;
	cudaStat = cudaMalloc((void**)&d_Array, sizeof(float) * cols);
	assert(cudaSuccess == cudaStat);
	float * d_Result = NULL;  // W = S*VT
	cudaStat = cudaMalloc((void**)&d_Result, sizeof(float) * rows * cols);
	assert(cudaSuccess == cudaStat);

	cudaStat = cudaMemcpy(d_Matrix, matrix, sizeof(float) * rows * cols, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat);
	cudaStat = cudaMemcpy(d_Array, array, sizeof(float) * cols, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat);

	row_major_2_col_major_device(rows, cols, d_Matrix);

	// create cublas handle
	cublasHandle_t cublasH = NULL;
	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
	cublas_status = cublasCreate(&cublasH);
	assert(CUBLAS_STATUS_SUCCESS == cublas_status);

	// R = M*S
	cublas_status = cublasSdgmm(
		cublasH,
		CUBLAS_SIDE_RIGHT,
		rows,
		cols,
		d_Matrix,
		rows,
		d_Array,
		1,
		d_Result,
		rows);
	assert(CUBLAS_STATUS_SUCCESS == cublas_status);

	col_major_2_row_major_device(rows, cols, d_Result);

	// copy
	cudaStat = cudaMemcpy(matrix, d_Result, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost);
	assert(cudaSuccess == cudaStat);

	// free
	if (d_Array) cudaFree(d_Array);
	if (d_Matrix) cudaFree(d_Matrix);
	if (cublasH) cublasDestroy(cublasH);

	cudaDeviceReset();
}

void multiply(int rows, int k, int cols, float * matrixA, float * matrixB, float * result) {
	// copy
	cudaError_t cudaStat = cudaSuccess;
	float *d_A = NULL;
	cudaStat = cudaMalloc((void**)&d_A, sizeof(float) * rows * k);
	assert(cudaSuccess == cudaStat);
	float * d_B = NULL;
	cudaStat = cudaMalloc((void**)&d_B, sizeof(float) * k * cols);
	assert(cudaSuccess == cudaStat);
	float * d_Result = NULL;
	cudaStat = cudaMalloc((void**)&d_Result, sizeof(float) * rows * cols);
	assert(cudaSuccess == cudaStat);

	cudaStat = cudaMemcpy(d_A, matrixA, sizeof(float) * rows * k, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat);
	cudaStat = cudaMemcpy(d_B, matrixB, sizeof(float) * k * cols, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat);

	row_major_2_col_major_device(rows, k, d_A);
	row_major_2_col_major_device(k, cols, d_B);

	// create cublas handle
	cublasHandle_t cublasH = NULL;
	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
	cublas_status = cublasCreate(&cublasH);
	assert(CUBLAS_STATUS_SUCCESS == cublas_status);

	// A := U*W
	assert(cudaSuccess == cudaStat);
	const float h_one = 1;
	const float h_zero = 0;
	cublas_status = cublasSgemm_v2(
		cublasH,
		CUBLAS_OP_N,
		CUBLAS_OP_N,
		rows,
		cols,
		k,
		&h_one,
		d_A,
		rows,
		d_B,
		k,
		&h_zero,
		d_Result,
		rows);
	assert(CUBLAS_STATUS_SUCCESS == cublas_status);

	// copy
	col_major_2_row_major_device(rows, cols, d_Result);

	cudaStat = cudaMemcpy(result, d_Result, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost);
	assert(cudaSuccess == cudaStat);

	// free
	if (d_A) cudaFree(d_A);
	if (d_B) cudaFree(d_B);
	if (d_Result) cudaFree(d_Result);
	if (cublasH) cublasDestroy(cublasH);

	cudaDeviceReset();
}