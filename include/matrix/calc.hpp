#ifndef SANAE_NEURALNETWORK_MATRIX_CALC
#define SANAE_NEURALNETWORK_MATRIX_CALC

#include <type_traits>
#include "matrix.h"

template<typename T, bool RowMajor, typename Container, typename En>
inline Matrix<T,RowMajor,Container,En>& Matrix<T, RowMajor, Container, En>::add(const Matrix& other)
{
	if (this->_rows != other._rows || this->_cols != other._cols) {
		throw std::invalid_argument("Matrix dimensions must agree for addition.");
	}
	for (size_t i = 0; i < this->_data.size(); i++) {
		this->_data[i] += other._data[i];
	}
	return *this;
}
template<typename T, bool RowMajor, typename Container, typename En>
inline Matrix<T, RowMajor, Container, En>& Matrix<T, RowMajor, Container, En>::add(const Matrix<T,!RowMajor>& other)
{
	if (this->_rows != other._rows || this->_cols != other._cols) {
		throw std::invalid_argument("Matrix dimensions must agree for addition.");
	}
	for (size_t i = 0; i < this->_rows; i++) {
		for (size_t j = 0; j < this->_cols; j++) {
			(*this)(i, j) += other(i, j);
		}
	}
	return *this;
}

#endif