#include "matrix/matrix"
#include <chrono>
#include <iostream>
#include <random>

template<typename func>
void benchmark(const std::string& testName, func f) {
	auto start = std::chrono::high_resolution_clock::now();
	f();
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration = end - start;
	std::cout << testName << " took " << duration.count() << " ms\n";
}

constexpr size_t MATRIX_SIZE = 1000;
using Type = double;

static void benchmark_exec() {
	std::random_device seedgen;
	std::default_random_engine engine(seedgen());
	std::uniform_real_distribution<double> dist(0,1);

	Matrix<Type,false> matA(MATRIX_SIZE, MATRIX_SIZE, [&]() { return dist(engine); });
	Matrix<Type,false> matB(MATRIX_SIZE, MATRIX_SIZE, [&]() { return dist(engine); });

	std::cout << "Matrix Size: " << MATRIX_SIZE << "x" << MATRIX_SIZE << "\n" << std::endl;

#ifdef USE_OPENBLAS
		openblas_set_num_threads(std::thread::hardware_concurrency());
		std::cout << "OPENBLAS-Config:" << openblas_get_config() << std::endl;
		std::cout << "OPENBLAS-THREADS:" << openblas_get_num_threads() << "\n" << std::endl;
#endif

	// 加算
	benchmark("Addition", [&]() {
		matA.add(matB);
		});

	// 減算
	benchmark("Subtraction", [&]() {
		matA.sub(matB);
		});

	// アダマール積
	benchmark("Hadamard Multiplication", [&]() {
		matA.hadamard_mul(matB);
		});

	// スカラー乗算
	benchmark("Scalar Multiplication", [&]() {
		matA.scalar_mul(2.0);
		});

	// アダマール除算
	benchmark("Hadamard Division", [&]() {
		matA.hadamard_div(matB);
		});
	// スカラー除算
	benchmark("Scalar Division", [&]() {
		matA.scalar_div(2.0);
		});

	// 行列積
	benchmark("Matrix Multiplication", [&]() {
		matA.matrix_mul(matB);
		});

	// BLAS使用版
	if constexpr (can_use_blas<Type>::value) {
		std::cout << "\nBLAS Enabled Tests:\n" << std::endl;
		benchmark("Addition with BLAS", [&]() {
			matA.add<true>(matB);
			});
		benchmark("Subtraction with BLAS", [&]() {
			matA.sub<true>(matB);
			});
		benchmark("Scalar Multiplication with BLAS", [&]() {
			matA.scalar_mul<true>(2.0);
			});
		benchmark("Matrix Multiplication with BLAS", [&]() {
			matA.matrix_mul<true>(matB);
			});
	}
}