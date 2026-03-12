#ifndef SANAE_NEURALNETWORK_CLBLAST_GEMM
#define SANAE_NEURALNETWORK_CLBLAST_GEMM

#if defined(USE_CLBLAST)
#include <CL/opencl.hpp>
#include <clblast.h>

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

			cl::Context context(CL_DEVICE_TYPE_GPU);
			cl::CommandQueue queue(context);

			cl::Buffer bufA(context, CL_MEM_READ_ONLY, M * K * sizeof(float));
			cl::Buffer bufB(context, CL_MEM_READ_ONLY, K * N * sizeof(float));
			cl::Buffer bufC(context, CL_MEM_READ_WRITE, M * N * sizeof(float));

			queue.enqueueWriteBuffer(bufA, CL_TRUE, 0, M * K * sizeof(float), A);
			queue.enqueueWriteBuffer(bufB, CL_TRUE, 0, K * N * sizeof(float), B);

			auto layout = AMajor ? clblast::Layout::kRowMajor : clblast::Layout::kColMajor;
			auto transA = clblast::Transpose::kNo;
			auto transB = (AMajor == BMajor) ? clblast::Transpose::kNo : clblast::Transpose::kYes;

			size_t lda = AMajor ? K : M;
			size_t ldb = AMajor ? N : K;
			size_t ldc = AMajor ? N : M;

			auto status = clblast::Gemm<float>(
				layout, transA, transB,
				M, N, K,
				alpha,
				bufA(), 0, lda,
				bufB(), 0, ldb,
				beta,
				bufC(), 0, ldc,
				&queue()
			);

			if (status != clblast::StatusCode::kSuccess) {
				throw std::runtime_error("clblast::Gemm failed.");
			}

			queue.enqueueReadBuffer(bufC, CL_TRUE, 0, M * N * sizeof(float), C);
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

			cl::Context context(CL_DEVICE_TYPE_GPU);
			cl::CommandQueue queue(context);

			cl::Buffer bufA(context, CL_MEM_READ_ONLY, M * K * sizeof(double));
			cl::Buffer bufB(context, CL_MEM_READ_ONLY, K * N * sizeof(double));
			cl::Buffer bufC(context, CL_MEM_READ_WRITE, M * N * sizeof(double));

			queue.enqueueWriteBuffer(bufA, CL_TRUE, 0, M * K * sizeof(double), A);
			queue.enqueueWriteBuffer(bufB, CL_TRUE, 0, K * N * sizeof(double), B);

			auto layout = AMajor ? clblast::Layout::kRowMajor : clblast::Layout::kColMajor;
			auto transA = clblast::Transpose::kNo;
			auto transB = (AMajor == BMajor) ? clblast::Transpose::kNo : clblast::Transpose::kYes;

			size_t lda = AMajor ? K : M;
			size_t ldb = AMajor ? N : K;
			size_t ldc = AMajor ? N : M;

			auto status = clblast::Gemm<double>(
				layout, transA, transB,
				M, N, K,
				alpha,
				bufA(), 0, lda,
				bufB(), 0, ldb,
				beta,
				bufC(), 0, ldc,
				&queue()
			);

			if (status != clblast::StatusCode::kSuccess) {
				throw std::runtime_error("clblast::Gemm failed.");
			}

			queue.enqueueReadBuffer(bufC, CL_TRUE, 0, M * N * sizeof(double), C);
		}
	};
	template<typename T>
	struct Add {};
	template<>
	struct Add<float> {
		static void axpy(size_t n, float alpha, const float* x, float* y) {
			cl::Context context(CL_DEVICE_TYPE_GPU);
			cl::CommandQueue queue(context);

			cl::Buffer bufX(context, CL_MEM_READ_ONLY, n * sizeof(float));
			cl::Buffer bufY(context, CL_MEM_READ_WRITE, n * sizeof(float));

			queue.enqueueWriteBuffer(bufX, CL_TRUE, 0, n * sizeof(float), x);
			queue.enqueueWriteBuffer(bufY, CL_TRUE, 0, n * sizeof(float), y);

			auto status = clblast::Axpy<float>(
				n,
				alpha,
				bufX(), 0, 1,
				bufY(), 0, 1,
				&queue()
			);

			if (status != clblast::StatusCode::kSuccess) {
				throw std::runtime_error("clblast::Axpy failed.");
			}

			queue.enqueueReadBuffer(bufY, CL_TRUE, 0, n * sizeof(float), y);
		}
	};
	template<>
	struct Add<double> {
		static void axpy(size_t n, double alpha, const double* x, double* y) {
			cl::Context context(CL_DEVICE_TYPE_GPU);
			cl::CommandQueue queue(context);

			cl::Buffer bufX(context, CL_MEM_READ_ONLY, n * sizeof(double));
			cl::Buffer bufY(context, CL_MEM_READ_WRITE, n * sizeof(double));

			queue.enqueueWriteBuffer(bufX, CL_TRUE, 0, n * sizeof(double), x);
			queue.enqueueWriteBuffer(bufY, CL_TRUE, 0, n * sizeof(double), y);

			auto status = clblast::Axpy<double>(
				n,
				alpha,
				bufX(), 0, 1,
				bufY(), 0, 1,
				&queue()
			);

			if (status != clblast::StatusCode::kSuccess) {
				throw std::runtime_error("clblast::Axpy failed.");
			}

			queue.enqueueReadBuffer(bufY, CL_TRUE, 0, n * sizeof(double), y);
		}
	};
	template<typename T>
	struct Sub {};
	template<>
	struct Sub<float> {
		static void axpy(size_t n, float alpha, const float* x, float* y) {
			float negAlpha = -alpha;
			cl::Context context(CL_DEVICE_TYPE_GPU);
			cl::CommandQueue queue(context);

			cl::Buffer bufX(context, CL_MEM_READ_ONLY, n * sizeof(float));
			cl::Buffer bufY(context, CL_MEM_READ_WRITE, n * sizeof(float));

			queue.enqueueWriteBuffer(bufX, CL_TRUE, 0, n * sizeof(float), x);
			queue.enqueueWriteBuffer(bufY, CL_TRUE, 0, n * sizeof(float), y);

			auto status = clblast::Axpy<float>(
				n,
				negAlpha,
				bufX(), 0, 1,
				bufY(), 0, 1,
				&queue()
			);

			if (status != clblast::StatusCode::kSuccess) {
				throw std::runtime_error("clblast::Axpy failed.");
			}

			queue.enqueueReadBuffer(bufY, CL_TRUE, 0, n * sizeof(float), y);
		}
	};
	template<>
	struct Sub<double> {
		static void axpy(size_t n, double alpha, const double* x, double* y) {
			double negAlpha = -alpha;
			cl::Context context(CL_DEVICE_TYPE_GPU);
			cl::CommandQueue queue(context);

			cl::Buffer bufX(context, CL_MEM_READ_ONLY, n * sizeof(double));
			cl::Buffer bufY(context, CL_MEM_READ_WRITE, n * sizeof(double));

			queue.enqueueWriteBuffer(bufX, CL_TRUE, 0, n * sizeof(double), x);
			queue.enqueueWriteBuffer(bufY, CL_TRUE, 0, n * sizeof(double), y);

			auto status = clblast::Axpy<double>(
				n,
				negAlpha,
				bufX(), 0, 1,
				bufY(), 0, 1,
				&queue()
			);

			if (status != clblast::StatusCode::kSuccess) {
				throw std::runtime_error("clblast::Axpy failed.");
			}

			queue.enqueueReadBuffer(bufY, CL_TRUE, 0, n * sizeof(double), y);
		}
	};
	template<typename T>
	struct ScalarMul {};
	template<>
	struct ScalarMul<float> {
		static void scal(size_t n, float alpha, float* x) {
			cl::Context context(CL_DEVICE_TYPE_GPU);
			cl::CommandQueue queue(context);

			cl::Buffer bufX(context, CL_MEM_READ_WRITE, n * sizeof(float));

			queue.enqueueWriteBuffer(bufX, CL_TRUE, 0, n * sizeof(float), x);

			auto status = clblast::Scal<float>(
				n,
				alpha,
				bufX(), 0, 1,
				&queue()
			);

			if (status != clblast::StatusCode::kSuccess) {
				throw std::runtime_error("clblast::Scal failed.");
			}

			queue.enqueueReadBuffer(bufX, CL_TRUE, 0, n * sizeof(float), x);
		}
	};
	template<>
	struct ScalarMul<double> {
		static void scal(size_t n, double alpha, double* x) {
			cl::Context context(CL_DEVICE_TYPE_GPU);
			cl::CommandQueue queue(context);

			cl::Buffer bufX(context, CL_MEM_READ_WRITE, n * sizeof(double));

			queue.enqueueWriteBuffer(bufX, CL_TRUE, 0, n * sizeof(double), x);

			auto status = clblast::Scal<double>(
				n,
				alpha,
				bufX(), 0, 1,
				&queue()
			);

			if (status != clblast::StatusCode::kSuccess) {
				throw std::runtime_error("clblast::Scal failed.");
			}

			queue.enqueueReadBuffer(bufX, CL_TRUE, 0, n * sizeof(double), x);
		}
	};
}
#endif
#endif