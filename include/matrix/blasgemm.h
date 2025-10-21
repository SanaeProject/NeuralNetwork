#if !defined(SANAE_NEURALNETWORK_MATRIX_MATMUL) && defined(USE_OPENBLAS)
#define SANAE_NEURALNETWORK_MATRIX_MATMUL

#include "matrix.h"
#include <cblas.h>

namespace BlasGemm {
	template<typename T>
	struct MatMul {};
	template<>
	struct MatMul<float> {
		static void multiply(const float* A, const float* B, float* C, size_t M, size_t N, size_t K, bool rowMajor) {
			CBLAS_ORDER order = rowMajor ? CblasRowMajor : CblasColMajor;
			cblas_sgemm(order, CblasNoTrans, CblasNoTrans,
				M, N, K,
				1.0f,
				A, rowMajor ? K : M,
				B, rowMajor ? N : K,
				0.0f,
				C, rowMajor ? N : M);
		}
	};
	template<>
	struct MatMul<double> {
		static void multiply(const double* A, const double* B, double* C, size_t M, size_t N, size_t K, bool rowMajor) {
			CBLAS_ORDER order = rowMajor ? CblasRowMajor : CblasColMajor;
			cblas_dgemm(order, CblasNoTrans, CblasNoTrans,
				M, N, K,
				1.0,
				A, rowMajor ? K : M,
				B, rowMajor ? N : K,
				0.0,
				C, rowMajor ? N : M);
		}
	};

	template<typename T>
	struct add{};
	template<>
	struct add<float> {
		static void axpy(size_t n, float alpha, const float* x, float* y) {
			cblas_saxpy(static_cast<int>(n), alpha, x, 1, y, 1);
		}
	};
	template<>
	struct add<double> {
		static void axpy(size_t n, double alpha, const double* x, double* y) {
			cblas_daxpy(static_cast<int>(n), alpha, x, 1, y, 1);
		}
	};
	template<typename T>
	struct sub {};
	template<>
	struct sub<float> {
		static void axpy(size_t n, float alpha, const float* x, float* y) {
			cblas_saxpy(static_cast<int>(n), -alpha, x, 1, y, 1);
		}
	};
	template<>
	struct sub<double> {
		static void axpy(size_t n, double alpha, const double* x, double* y) {
			cblas_daxpy(static_cast<int>(n), -alpha, x, 1, y, 1);
		}
	};

	template<typename T>
	struct scalar_mul {};
	template<>
	struct scalar_mul<float> {
		static void scal(size_t n, float alpha, float* x) {
			cblas_sscal(static_cast<int>(n), alpha, x, 1);
		}
	};
	template<>
	struct scalar_mul<double> {
		static void scal(size_t n, double alpha, double* x) {
			cblas_dscal(static_cast<int>(n), alpha, x, 1);
		}
	};
}

#endif