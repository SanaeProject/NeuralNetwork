#ifndef SANAE_NEURALNETWORK_MATRIX_MATMUL
#define SANAE_NEURALNETWORK_MATRIX_MATMUL

#include "matrix.h"

#if defined(USE_OPENBLAS)
#include <cblas.h>

namespace BlasGemm {
	template<typename T>
	struct MatMul {};
	template<>
	struct MatMul<float> {
		static void multiply(const float* A, const float* B, float* C, size_t M, size_t N, size_t K, bool AMajor, bool BMajor) {
			CBLAS_ORDER order = AMajor ? CblasRowMajor : CblasColMajor;
			CBLAS_TRANSPOSE transA = CblasNoTrans;
			CBLAS_TRANSPOSE transB = AMajor == BMajor ? CblasNoTrans : CblasTrans;

			cblas_sgemm(order, transA, transB,
				M, N, K,
				1.0,
				A, AMajor ? K : M,
				B, AMajor ? N : K,
				0.0,
				C, AMajor ? N : M);
		}
	};
	template<>
	struct MatMul<double> {
		static void multiply(const double* A, const double* B, double* C, size_t M, size_t N, size_t K, bool AMajor, bool BMajor) {
			CBLAS_ORDER order = AMajor ? CblasRowMajor : CblasColMajor;
			CBLAS_TRANSPOSE transA = CblasNoTrans;
			CBLAS_TRANSPOSE transB = AMajor == BMajor ? CblasNoTrans : CblasTrans;

			cblas_dgemm(order, transA, transB,
				M, N, K,
				1.0,
				A, AMajor ? K : M,
				B, AMajor ? N : K,
				0.0,
				C, AMajor ? N : M);
		}
	};

	template<typename T>
	struct Add{};
	template<>
	struct Add<float> {
		static void axpy(size_t n, float alpha, const float* x, float* y) {
			cblas_saxpy(static_cast<int>(n), alpha, x, 1, y, 1);
		}
	};
	template<>
	struct Add<double> {
		static void axpy(size_t n, double alpha, const double* x, double* y) {
			cblas_daxpy(static_cast<int>(n), alpha, x, 1, y, 1);
		}
	};
	template<typename T>
	struct Sub {};
	template<>
	struct Sub<float> {
		static void axpy(size_t n, float alpha, const float* x, float* y) {
			cblas_saxpy(static_cast<int>(n), -alpha, x, 1, y, 1);
		}
	};
	template<>
	struct Sub<double> {
		static void axpy(size_t n, double alpha, const double* x, double* y) {
			cblas_daxpy(static_cast<int>(n), -alpha, x, 1, y, 1);
		}
	};

	template<typename T>
	struct ScalarMul {};
	template<>
	struct ScalarMul<float> {
		static void scal(size_t n, float alpha, float* x) {
			cblas_sscal(static_cast<int>(n), alpha, x, 1);
		}
	};
	template<>
	struct ScalarMul<double> {
		static void scal(size_t n, double alpha, double* x) {
			cblas_dscal(static_cast<int>(n), alpha, x, 1);
		}
	};
}
#elif defined(USE_CUBLAS)

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace BlasGemm {
	template<typename T>
	struct MatMul {};

	template<>
	struct MatMul<float> {
		static void multiply(const float* A, const float* B, float* C, size_t M, size_t N, size_t K, bool AMajor, bool BMajor) {
			float* dA, * dB, * dC;
			const float alpha = 1.0f, beta = 0.0f;

			cudaMalloc(&dA, M * K * sizeof(float));
			cudaMalloc(&dB, K * N * sizeof(float));
			cudaMalloc(&dC, M * N * sizeof(float));

			cudaMemcpy(dA, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(dB, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

			cublasHandle_t handle;
			cublasCreate(&handle);

			cublasOperation_t transA = AMajor ? CUBLAS_OP_T : CUBLAS_OP_N;
			int lda = AMajor ? K : M;   // Row-major → K, Col-major → M

			cublasOperation_t transB = BMajor ? CUBLAS_OP_T : CUBLAS_OP_N;
			int ldb = BMajor ? N : K;   // Row-major → N, Col-major → K

			// --- C は常に M×N の列メジャーとして扱う ---
			int ldc = M;

			cublasSgemm(handle, transA, transB,
				M, N, K,
				&alpha,
				dA, lda,
				dB, ldb,
				&beta,
				dC, ldc
			);

			cudaMemcpy(C, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);

			cublasDestroy(handle);
			cudaFree(dA); cudaFree(dB); cudaFree(dC);
		}
	};
	template<>
	struct MatMul<double> {
		static void multiply(const double* A, const double* B, double* C, size_t M, size_t N, size_t K, bool AMajor, bool BMajor) {
			double* dA, * dB, * dC;
			const double alpha = 1.0, beta = 0.0;

			cudaMalloc(&dA, M * K * sizeof(double));
			cudaMalloc(&dB, K * N * sizeof(double));
			cudaMalloc(&dC, M * N * sizeof(double));

			cudaMemcpy(dA, A, M * K * sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(dB, B, K * N * sizeof(double), cudaMemcpyHostToDevice);

			cublasHandle_t handle;
			cublasCreate(&handle);

			cublasOperation_t transA = AMajor ? CUBLAS_OP_T : CUBLAS_OP_N;
			int lda = AMajor ? K : M;

			cublasOperation_t transB = BMajor ? CUBLAS_OP_T : CUBLAS_OP_N;
			int ldb = BMajor ? N : K;

			int ldc = M;

			cublasDgemm( handle, transA, transB,
				M, N, K,
				&alpha,
				dA, lda,
				dB, ldb,
				&beta,
				dC, ldc
			);

			cudaMemcpy(C, dC, M * N * sizeof(double), cudaMemcpyDeviceToHost);

			cublasDestroy(handle);
			cudaFree(dA); cudaFree(dB); cudaFree(dC);
		}
	};
	template<typename T>
	struct Add {};
	template<>
	struct Add<float> {
		static void axpy(size_t n, float alpha, const float* x, float* y) {
			float* dx, * dy;
			cudaMalloc(&dx, n * sizeof(float));
			cudaMalloc(&dy, n * sizeof(float));
			cudaMemcpy(dx, x, n * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(dy, y, n * sizeof(float), cudaMemcpyHostToDevice);

			cublasHandle_t handle;
			cublasCreate(&handle);
			cublasSaxpy(handle, static_cast<int>(n), &alpha, dx, 1, dy, 1);
			cudaMemcpy(y, dy, n * sizeof(float), cudaMemcpyDeviceToHost);

			cublasDestroy(handle);
			cudaFree(dx); cudaFree(dy);
		}
	};
	template<>
	struct Add<double> {
		static void axpy(size_t n, double alpha, const double* x, double* y) {
			double* dx, * dy;
			cudaMalloc(&dx, n * sizeof(double));
			cudaMalloc(&dy, n * sizeof(double));
			cudaMemcpy(dx, x, n * sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(dy, y, n * sizeof(double), cudaMemcpyHostToDevice);

			cublasHandle_t handle;
			cublasCreate(&handle);
			cublasDaxpy(handle, static_cast<int>(n), &alpha, dx, 1, dy, 1);
			cudaMemcpy(y, dy, n * sizeof(double), cudaMemcpyDeviceToHost);

			cublasDestroy(handle);
			cudaFree(dx); cudaFree(dy);
		}
	};
	template<typename T>
	struct Sub {};
	template<>
	struct Sub<float> {
		static void axpy(size_t n, float alpha, const float* x, float* y) {
			float negAlpha = -alpha;
			float* dx, * dy;
			cudaMalloc(&dx, n * sizeof(float));
			cudaMalloc(&dy, n * sizeof(float));
			cudaMemcpy(dx, x, n * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(dy, y, n * sizeof(float), cudaMemcpyHostToDevice);

			cublasHandle_t handle;
			cublasCreate(&handle);
			cublasSaxpy(handle, static_cast<int>(n), &negAlpha, dx, 1, dy, 1);
			cudaMemcpy(y, dy, n * sizeof(float), cudaMemcpyDeviceToHost);

			cublasDestroy(handle);
			cudaFree(dx); cudaFree(dy);
		}
	};
	template<>
	struct Sub<double> {
		static void axpy(size_t n, double alpha, const double* x, double* y) {
			double negAlpha = -alpha;
			double* dx, * dy;
			cudaMalloc(&dx, n * sizeof(double));
			cudaMalloc(&dy, n * sizeof(double));
			cudaMemcpy(dx, x, n * sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(dy, y, n * sizeof(double), cudaMemcpyHostToDevice);

			cublasHandle_t handle;
			cublasCreate(&handle);
			cublasDaxpy(handle, static_cast<int>(n), &negAlpha, dx, 1, dy, 1);
			cudaMemcpy(y, dy, n * sizeof(double), cudaMemcpyDeviceToHost);

			cublasDestroy(handle);
			cudaFree(dx); cudaFree(dy);
		}
	};
	template<typename T>
	struct ScalarMul {};
	template<>
	struct ScalarMul<float> {
		static void scal(size_t n, float alpha, float* x) {
			float* dx;
			cudaMalloc(&dx, n * sizeof(float));
			cudaMemcpy(dx, x, n * sizeof(float), cudaMemcpyHostToDevice);

			cublasHandle_t handle;
			cublasCreate(&handle);
			cublasSscal(handle, static_cast<int>(n), &alpha, dx, 1);
			cudaMemcpy(x, dx, n * sizeof(float), cudaMemcpyDeviceToHost);

			cublasDestroy(handle);
			cudaFree(dx);
		}
	};
	template<>
	struct ScalarMul<double> {
		static void scal(size_t n, double alpha, double* x) {
			double* dx;
			cudaMalloc(&dx, n * sizeof(double));
			cudaMemcpy(dx, x, n * sizeof(double), cudaMemcpyHostToDevice);

			cublasHandle_t handle;
			cublasCreate(&handle);
			cublasDscal(handle, static_cast<int>(n), &alpha, dx, 1);
			cudaMemcpy(x, dx, n * sizeof(double), cudaMemcpyDeviceToHost);

			cublasDestroy(handle);
			cudaFree(dx);
		}
	};
}
	
#else
namespace BlasGemm {
	template<typename T> 
	struct MatMul {
		static void multiply(const T* A, const T* B, T* C, size_t M, size_t N, size_t K, bool rowMajor) {
			// BLAS未使用時のプレースホルダ
			throw std::runtime_error("BLAS not supported for this data type.");
		}
	};
	template<typename T> 
	struct Add {
		static void axpy(size_t n, T alpha, const T* x, T* y) {
			// BLAS未使用時のプレースホルダ
			throw std::runtime_error("BLAS not supported for this data type.");
		}
	};
	template<typename T> 
	struct Sub {
		static void axpy(size_t n, T alpha, const T* x, T* y) {
			// BLAS未使用時のプレースホルダ
			throw std::runtime_error("BLAS not supported for this data type.");
		}
	};
	template<typename T> 
	struct ScalarMul {
		static void scal(size_t n, T alpha, T* x) {
			// BLAS未使用時のプレースホルダ
			throw std::runtime_error("BLAS not supported for this data type.");
		}
	};
}

#endif

#endif