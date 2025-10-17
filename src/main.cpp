#include <cassert>
#include <iostream>
#include <matrix/matrix>
#include <vector>

int main() {
	// Row major matrix [1,2,3,4,5,6]
	Matrix<float> matA = {
		{1.0f, 2.0f, 3.0f},
		{4.0f, 5.0f, 6.0f}
	};
	for (size_t i = 0; i < matA.rows(); ++i) {
		for (size_t j = 0; j < matA.cols(); ++j) {
			std::cout << matA(i, j) << " ";
		}
		std::cout << std::endl;
	}

	// Column major matrix [1,4,2,5,3,6]
	Matrix<float, false> matB = {
		{1.0f, 2.0f, 3.0f},
		{4.0f, 5.0f, 6.0f}
	};
	for (size_t i = 0; i < matB.rows(); ++i) {
		for (size_t j = 0; j < matB.cols(); ++j) {
			std::cout << matB(i, j) << " ";
		}
		std::cout << std::endl;
	}

	// Assertion check!
	assert(matB == matA);

	matB = matA.convertLayout();

	// Assertion check!
	assert(matB==matA);
}
