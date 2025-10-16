#ifndef SANAE_NEURALNETWORK_MATRIX_UTIL
#define SANAE_NEURALNETWORK_MATRIX_UTIL

#include <type_traits>
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


#endif // SANAE_NEURALNETWORK_MATRIX_UTIL