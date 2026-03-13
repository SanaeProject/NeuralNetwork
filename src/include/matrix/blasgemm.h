#ifndef SANAE_NEURALNETWORK_MATRIX_MATMUL
#define SANAE_NEURALNETWORK_MATRIX_MATMUL

#include "./blasgemms/clblast-gemm.hpp"
#include "./blasgemms/cublas-gemm.hpp"
#include "./blasgemms/openblas-gemm.hpp"

// BLASライブラリを一切使用しない場合のプレースホルダ
#ifndef USE_BLAS
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