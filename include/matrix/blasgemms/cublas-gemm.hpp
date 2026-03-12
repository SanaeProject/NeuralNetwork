#ifndef SANAE_NEURALNETWORK_CUBLAS_GEMM
#define SANAE_NEURALNETWORK_CUBLAS_GEMM

#if defined(USE_CUBLAS)

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace BlasGemm {
	template<typename T>
	struct MatMul {};

	template<>
	struct MatMul<float> {
		static void multiply(
			const float* A, const float* B, float* C,
			size_t M, size_t N, size_t K,
			bool AMajor, bool BMajor
		) {
			const float alpha = 1.0f;
			const float beta = 0.0f;

			float* dA = nullptr, * dB = nullptr, * dC = nullptr;

			auto cleanup = [&]() {
				if (dA) cudaFree(dA);
				if (dB) cudaFree(dB);
				if (dC) cudaFree(dC);
				};

			if (cudaMalloc(&dA, M * K * sizeof(float)) != cudaSuccess)
				throw std::runtime_error("Failed to allocate device memory for A.");

			if (cudaMalloc(&dB, K * N * sizeof(float)) != cudaSuccess) {
				cleanup();
				throw std::runtime_error("Failed to allocate device memory for B.");
			}

			if (cudaMalloc(&dC, M * N * sizeof(float)) != cudaSuccess) {
				cleanup();
				throw std::runtime_error("Failed to allocate device memory for C.");
			}

			if (cudaMemcpy(dA, A, M * K * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
				cleanup();
				throw std::runtime_error("Failed to copy A to device.");
			}

			if (cudaMemcpy(dB, B, K * N * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
				cleanup();
				throw std::runtime_error("Failed to copy B to device.");
			}

			cublasHandle_t handle;
			cublasCreate(&handle);

			cublasOperation_t transA = AMajor ? CUBLAS_OP_N : CUBLAS_OP_T;
			cublasOperation_t transB = BMajor ? CUBLAS_OP_N : CUBLAS_OP_T;

			int lda = AMajor ? K : M;
			int ldb = BMajor ? N : K;
			int ldc = N;

			cublasStatus_t status = cublasSgemm(
				handle,
				transB, transA,   // cuBLAS は列優先なので順序が逆になる
				N, M, K,
				&alpha,
				dB, ldb,
				dA, lda,
				&beta,
				dC, ldc
			);

			if (status != CUBLAS_STATUS_SUCCESS) {
				cublasDestroy(handle);
				cleanup();
				throw std::runtime_error("cublasSgemm failed.");
			}

			if (cudaMemcpy(C, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
				cublasDestroy(handle);
				cleanup();
				throw std::runtime_error("Failed to copy C from device.");
			}

			cublasDestroy(handle);
			cleanup();
		}
	};
	template<>
	struct MatMul<double> {
		static void multiply(
			const double* A, const double* B, double* C,
			size_t M, size_t N, size_t K,
			bool AMajor, bool BMajor
		) {
			const double alpha = 1.0;
			const double beta = 0.0;

			double* dA = nullptr, * dB = nullptr, * dC = nullptr;

			auto cleanup = [&]() {
				if (dA) cudaFree(dA);
				if (dB) cudaFree(dB);
				if (dC) cudaFree(dC);
				};

			if (cudaMalloc(&dA, M * K * sizeof(double)) != cudaSuccess)
				throw std::runtime_error("Failed to allocate device memory for A.");

			if (cudaMalloc(&dB, K * N * sizeof(double)) != cudaSuccess) {
				cleanup();
				throw std::runtime_error("Failed to allocate device memory for B.");
			}

			if (cudaMalloc(&dC, M * N * sizeof(double)) != cudaSuccess) {
				cleanup();
				throw std::runtime_error("Failed to allocate device memory for C.");
			}

			if (cudaMemcpy(dA, A, M * K * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
				cleanup();
				throw std::runtime_error("Failed to copy A to device.");
			}

			if (cudaMemcpy(dB, B, K * N * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
				cleanup();
				throw std::runtime_error("Failed to copy B to device.");
			}

			cublasHandle_t handle;
			cublasCreate(&handle);

			cublasOperation_t transA = AMajor ? CUBLAS_OP_N : CUBLAS_OP_T;
			cublasOperation_t transB = BMajor ? CUBLAS_OP_N : CUBLAS_OP_T;

			int lda = AMajor ? K : M;
			int ldb = BMajor ? N : K;
			int ldc = N;

			cublasStatus_t status = cublasDgemm(
				handle,
				transB, transA,   // cuBLAS は列優先なので A/B の順序が逆になる
				N, M, K,
				&alpha,
				dB, ldb,
				dA, lda,
				&beta,
				dC, ldc
			);

			if (status != CUBLAS_STATUS_SUCCESS) {
				cublasDestroy(handle);
				cleanup();
				throw std::runtime_error("cublasDgemm failed.");
			}

			if (cudaMemcpy(C, dC, M * N * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
				cublasDestroy(handle);
				cleanup();
				throw std::runtime_error("Failed to copy C from device.");
			}

			cublasDestroy(handle);
			cleanup();
		}
	};
	template<typename T>
	struct Add {};
	template<>
	struct Add<float> {
		static void axpy(size_t n, float alpha, const float* x, float* y) {
			float* dx = nullptr, * dy = nullptr;

			auto cleanup = [&]() {
				if (dx) cudaFree(dx);
				if (dy) cudaFree(dy);
				};

			if (cudaMalloc(&dx, n * sizeof(float)) != cudaSuccess)
				throw std::runtime_error("Failed to allocate device memory for x.");

			if (cudaMalloc(&dy, n * sizeof(float)) != cudaSuccess) {
				cleanup();
				throw std::runtime_error("Failed to allocate device memory for y.");
			}

			if (cudaMemcpy(dx, x, n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
				cleanup();
				throw std::runtime_error("Failed to copy x to device.");
			}

			if (cudaMemcpy(dy, y, n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
				cleanup();
				throw std::runtime_error("Failed to copy y to device.");
			}

			cublasHandle_t handle;
			cublasCreate(&handle);

			cublasStatus_t status = cublasSaxpy(
				handle,
				static_cast<int>(n),
				&alpha,
				dx, 1,
				dy, 1
			);

			if (status != CUBLAS_STATUS_SUCCESS) {
				cublasDestroy(handle);
				cleanup();
				throw std::runtime_error("cublasSaxpy failed.");
			}

			if (cudaMemcpy(y, dy, n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
				cublasDestroy(handle);
				cleanup();
				throw std::runtime_error("Failed to copy y from device.");
			}

			cublasDestroy(handle);
			cleanup();
		}
	};
	template<>
	struct Add<double> {
		static void axpy(size_t n, double alpha, const double* x, double* y) {
			double* dx = nullptr, * dy = nullptr;

			auto cleanup = [&]() {
				if (dx) cudaFree(dx);
				if (dy) cudaFree(dy);
				};

			if (cudaMalloc(&dx, n * sizeof(double)) != cudaSuccess)
				throw std::runtime_error("Failed to allocate dx.");

			if (cudaMalloc(&dy, n * sizeof(double)) != cudaSuccess) {
				cleanup();
				throw std::runtime_error("Failed to allocate dy.");
			}

			if (cudaMemcpy(dx, x, n * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
				cudaMemcpy(dy, y, n * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
				cleanup();
				throw std::runtime_error("Failed to copy x or y to device.");
			}

			cublasHandle_t handle;
			cublasCreate(&handle);

			cublasStatus_t status = cublasDaxpy(
				handle, static_cast<int>(n),
				&alpha, dx, 1, dy, 1
			);

			if (status != CUBLAS_STATUS_SUCCESS) {
				cublasDestroy(handle);
				cleanup();
				throw std::runtime_error("cublasDaxpy failed.");
			}

			if (cudaMemcpy(y, dy, n * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
				cublasDestroy(handle);
				cleanup();
				throw std::runtime_error("Failed to copy y from device.");
			}

			cublasDestroy(handle);
			cleanup();
		}
	};
	template<typename T>
	struct Sub {};
	template<>
	struct Sub<float> {
		static void axpy(size_t n, float alpha, const float* x, float* y) {
			float negAlpha = -alpha;
			float* dx = nullptr, * dy = nullptr;

			auto cleanup = [&]() {
				if (dx) cudaFree(dx);
				if (dy) cudaFree(dy);
				};

			if (cudaMalloc(&dx, n * sizeof(float)) != cudaSuccess)
				throw std::runtime_error("Failed to allocate dx.");

			if (cudaMalloc(&dy, n * sizeof(float)) != cudaSuccess) {
				cleanup();
				throw std::runtime_error("Failed to allocate dy.");
			}

			if (cudaMemcpy(dx, x, n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
				cudaMemcpy(dy, y, n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
				cleanup();
				throw std::runtime_error("Failed to copy x or y to device.");
			}

			cublasHandle_t handle;
			cublasCreate(&handle);

			cublasStatus_t status = cublasSaxpy(
				handle, static_cast<int>(n),
				&negAlpha, dx, 1, dy, 1
			);

			if (status != CUBLAS_STATUS_SUCCESS) {
				cublasDestroy(handle);
				cleanup();
				throw std::runtime_error("cublasSaxpy failed.");
			}

			if (cudaMemcpy(y, dy, n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
				cublasDestroy(handle);
				cleanup();
				throw std::runtime_error("Failed to copy y from device.");
			}

			cublasDestroy(handle);
			cleanup();
		}
	};
	template<>
	struct Sub<double> {
		static void axpy(size_t n, double alpha, const double* x, double* y) {
			double negAlpha = -alpha;
			double* dx = nullptr, * dy = nullptr;

			auto cleanup = [&]() {
				if (dx) cudaFree(dx);
				if (dy) cudaFree(dy);
				};

			if (cudaMalloc(&dx, n * sizeof(double)) != cudaSuccess)
				throw std::runtime_error("Failed to allocate dx.");

			if (cudaMalloc(&dy, n * sizeof(double)) != cudaSuccess) {
				cleanup();
				throw std::runtime_error("Failed to allocate dy.");
			}

			if (cudaMemcpy(dx, x, n * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
				cudaMemcpy(dy, y, n * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
				cleanup();
				throw std::runtime_error("Failed to copy x or y to device.");
			}

			cublasHandle_t handle;
			cublasCreate(&handle);

			cublasStatus_t status = cublasDaxpy(
				handle, static_cast<int>(n),
				&negAlpha, dx, 1, dy, 1
			);

			if (status != CUBLAS_STATUS_SUCCESS) {
				cublasDestroy(handle);
				cleanup();
				throw std::runtime_error("cublasDaxpy failed.");
			}

			if (cudaMemcpy(y, dy, n * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
				cublasDestroy(handle);
				cleanup();
				throw std::runtime_error("Failed to copy y from device.");
			}

			cublasDestroy(handle);
			cleanup();
		}
	};
	template<typename T>
	struct ScalarMul {};
	template<>
	struct ScalarMul<float> {
		static void scal(size_t n, float alpha, float* x) {
			float* dx = nullptr;

			auto cleanup = [&]() {
				if (dx) cudaFree(dx);
				};

			if (cudaMalloc(&dx, n * sizeof(float)) != cudaSuccess)
				throw std::runtime_error("Failed to allocate dx.");

			if (cudaMemcpy(dx, x, n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
				cleanup();
				throw std::runtime_error("Failed to copy x to device.");
			}

			cublasHandle_t handle;
			cublasCreate(&handle);

			cublasStatus_t status = cublasSscal(
				handle, static_cast<int>(n),
				&alpha, dx, 1
			);

			if (status != CUBLAS_STATUS_SUCCESS) {
				cublasDestroy(handle);
				cleanup();
				throw std::runtime_error("cublasSscal failed.");
			}

			if (cudaMemcpy(x, dx, n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
				cublasDestroy(handle);
				cleanup();
				throw std::runtime_error("Failed to copy x from device.");
			}

			cublasDestroy(handle);
			cleanup();
		}
	};
	template<>
	struct ScalarMul<double> {
		static void scal(size_t n, double alpha, double* x) {
			double* dx = nullptr;

			auto cleanup = [&]() {
				if (dx) cudaFree(dx);
				};

			if (cudaMalloc(&dx, n * sizeof(double)) != cudaSuccess)
				throw std::runtime_error("Failed to allocate dx.");

			if (cudaMemcpy(dx, x, n * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
				cleanup();
				throw std::runtime_error("Failed to copy x to device.");
			}

			cublasHandle_t handle;
			cublasCreate(&handle);

			cublasStatus_t status = cublasDscal(
				handle, static_cast<int>(n),
				&alpha, dx, 1
			);

			if (status != CUBLAS_STATUS_SUCCESS) {
				cublasDestroy(handle);
				cleanup();
				throw std::runtime_error("cublasDscal failed.");
			}

			if (cudaMemcpy(x, dx, n * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
				cublasDestroy(handle);
				cleanup();
				throw std::runtime_error("Failed to copy x from device.");
			}

			cublasDestroy(handle);
			cleanup();
		}
	};
}

#endif
#endif