#ifndef SANAE_NEURALNETWORK_MATRIX_OPS  
#define SANAE_NEURALNETWORK_MATRIX_OPS  

#include "matrix.h"  
#include <type_traits>
#include <utility>


template<typename T, bool RowMajor, typename Container, typename En>
inline T& Matrix<T, RowMajor, Container, En>::operator()(size_t row, size_t col)
{  
	if constexpr (!RowMajor)
		return this->_data[col * this->_rows + row];

	return this->_data[row * this->_cols + col];
}

template<typename T, bool RowMajor, typename Container, typename En>
inline T& Matrix<T, RowMajor, Container, En>::operator()(size_t index)
{
	return this->_data[index];
}


#endif // SANAE_NEURALNETWORK_MATRIX_OPS