#ifndef SANAE_NEURALNETWORK_MATRIX_OPS  
#define SANAE_NEURALNETWORK_MATRIX_OPS  

#include "matrix.h"  
#include <iosfwd>
#include <ostream>

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
template<typename T, bool RowMajor, typename Container, typename En>
inline T& Matrix<T, RowMajor, Container, En>::operator[](size_t index)
{
	return this->_data[index];
}
template<typename T, bool RowMajor, typename Container, typename En>
inline const T& Matrix<T, RowMajor, Container, En>::operator()(size_t row, size_t col) const
{
	if constexpr (!RowMajor)
		return this->_data[col * this->_rows + row];

	return this->_data[row * this->_cols + col];
}
template<typename T, bool RowMajor, typename Container, typename En>
inline const T& Matrix<T, RowMajor, Container, En>::operator()(size_t index) const
{
	return this->_data[index];
}
template<typename T, bool RowMajor, typename Container, typename En>
inline const T& Matrix<T, RowMajor, Container, En>::operator[](size_t index) const
{
	return this->_data[index];
}
template<typename T, bool RowMajor, typename Container, typename En>
inline bool Matrix<T, RowMajor, Container, En>::operator==(const Matrix& other) const
{
	if (this->cols() != other.cols() || this->rows() != other.rows())
		return false;

	const size_t total_elements = this->rows() * this->cols();
	for (size_t i = 0; i < total_elements; ++i) {
		if (this->_data[i] != other._data[i])
			return false;
	}

	return true;
}
template<typename T, bool RowMajor, typename Container, typename En>
inline bool Matrix<T, RowMajor, Container, En>::operator!=(const Matrix& other) const
{
	return !(*this == other);
}
template<typename T, bool RowMajor, typename Container, typename En>
inline bool Matrix<T, RowMajor, Container, En>::operator==(const Matrix<T,!RowMajor>& other) const
{
	if (this->cols() != other.cols() || this->rows() != other.rows())
		return false;
	const size_t rows = this->rows();
	const size_t cols = this->cols();

	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j) {
			if (this->operator()(i, j) != other(i, j))
				return false;
		}
	}

	return true;
}
template<typename T, bool RowMajor, typename Container, typename En>
inline bool Matrix<T, RowMajor, Container, En>::operator!=(const Matrix<T,!RowMajor>& other) const
{
	return !(*this == other);
}
template<typename T, bool RowMajor, typename Container, typename En>
std::ostream& operator<<(std::ostream& os, const Matrix<T, RowMajor, Container, En>& mat)
{
	const size_t rows = mat.rows();
	const size_t cols = mat.cols();

	auto comma_if_not_last = [](size_t idx, size_t total) -> const char* {
		return (idx + 1 != total) ? "," : "";
	};
	os << "{";
	for (size_t i = 0; i < rows; i++) {
		os << "{";
		for (size_t j = 0; j < cols; j++) {
			os << mat(i, j) << comma_if_not_last(j, cols);
		}
		os << "}" << comma_if_not_last(i, rows);
	}
	os << "}" ;

	return os;
}

#endif // SANAE_NEURALNETWORK_MATRIX_OPS