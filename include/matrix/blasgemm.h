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