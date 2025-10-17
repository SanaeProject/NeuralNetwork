#ifndef SANAE_NEURALNETWORK_MATRIX_UTIL
#define SANAE_NEURALNETWORK_MATRIX_UTIL

#include "matrix.h"

template<typename T, bool RowMajor, typename Container, typename En>
inline size_t Matrix<T, RowMajor, Container, En>::rows() const
{
	return this->_rows;
}
template<typename T, bool RowMajor, typename Container, typename En>
inline size_t Matrix<T, RowMajor, Container, En>::cols() const
{
	return this->_cols;
}
template<typename T, bool RowMajor, typename Container, typename En>
inline const Container& Matrix<T, RowMajor, Container, En>::data() const
{
	return this->_data;
}
template<typename T, bool RowMajor, typename Container, typename En>
inline Matrix<T, !RowMajor> Matrix<T, RowMajor, Container, En>::convertLayout() const
{
	Matrix<T, !RowMajor> result(this->rows(), this->cols());
	for (size_t i = 0; i < this->rows(); i++) {
		for (size_t j = 0; j < this->cols(); j++) {
			result(i, j) = this->operator()(i, j);
		}
	}
	return result;
}

#endif // SANAE_NEURALNETWORK_MATRIX_UTIL