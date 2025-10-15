#include <iostream>
#include <vector>
#include <cblas.h>

int main() {
	std::vector<float> mtxA = { 1.0, 2.0, 3.0, 4.0 };
	std::vector<float> mtxB = { 5.0, 6.0, 7.0, 8.0 };

	std::vector<float> mtxC(4, 0.0);
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2, 2, 2, 1.0, mtxA.data(), 2, mtxB.data(), 2, 0.0, mtxC.data(), 2);
    
	for (const auto& val : mtxC) {
		std::cout << val << " ";
	}
    return 0;
}
