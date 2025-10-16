#include <iostream>
#include <vector>
#include <cblas.h>
#include <matrix/matrix>

int main() {
	// çsóDêÊ
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

	// óÒóDêÊ
	Matrix<float, false> matB = {
		{1.0f, 2.0f},
		{3.0f, 4.0f},
		{5.0f, 6.0f}
	};
	for (size_t i = 0; i < matB.rows(); ++i) {
		for (size_t j = 0; j < matB.cols(); ++j) {
			std::cout << matB(i, j) << " ";
		}
		std::cout << std::endl;
	}

}
