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

constexpr size_t MATRIX_SIZE = 10000;
using Type = double;

static void exec() {
	std::random_device seedgen;
	std::default_random_engine engine(seedgen());
	std::uniform_real_distribution<double> dist(0,1);

	Matrix<Type> matA(MATRIX_SIZE, MATRIX_SIZE, dist(engine));
	Matrix<Type> matB(MATRIX_SIZE, MATRIX_SIZE, dist(engine));

	/*
	benchmark("Addition", [&]() {
		matA.add(matB);
		});
	if constexpr (can_use_blas<Type>::value) {
		benchmark("Addition with BLAS", [&]() {
			matA.add<true>(matB);
			});
	}

	benchmark("Subtraction", [&]() {
		matA.sub(matB);
		});

	if constexpr (can_use_blas<Type>::value) {
		benchmark("Subtraction with BLAS", [&]() {
			matA.sub<true>(matB);
			});
	}

	benchmark("Hadamard Multiplication", [&]() {
		matA.hadamard_mul(matB);
		});

	benchmark("Scalar Multiplication", [&]() {
		matA.scalar_mul(2.0);
		});

	if constexpr (can_use_blas<Type>::value) {
		benchmark("Scalar Multiplication with BLAS", [&]() {
			matA.scalar_mul<true>(2.0);
			});
	}

	benchmark("Hadamard Division", [&]() {
		matA.hadamard_div(matB);
		});

	benchmark("Scalar Division", [&]() {
		matA.scalar_div(2.0);
		});
		*/
	benchmark("Matrix Multiplication", [&]() {
		matA.matrix_mul<true>(matB);
		});
		
	if constexpr (can_use_blas<Type>::value) {
		benchmark("Matrix Multiplication with BLAS", [&]() {
			matA.matrix_mul<true>(matB);
			});
	}
}