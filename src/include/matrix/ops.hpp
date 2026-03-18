#ifndef SANAE_NEURALNETWORK_MATRIX_OPS  
#define SANAE_NEURALNETWORK_MATRIX_OPS  

#include "matrix.h"  
#include <iosfwd>
#include <ostream>

template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
inline T& Matrix<T, RowMajor, Container>::operator()(size_t row, size_t col)
{  
	if constexpr (!RowMajor)
		return this->_data[col * this->_rows + row];

	return this->_data[row * this->_cols + col];
}
template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
inline T& Matrix<T, RowMajor, Container>::operator()(size_t index)
{
	return this->_data[index];
}
template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
inline T& Matrix<T, RowMajor, Container>::operator[](size_t index)
{
	return this->_data[index];
}
template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
inline const T& Matrix<T, RowMajor, Container>::operator()(size_t row, size_t col) const
{
	if constexpr (!RowMajor)
		return this->_data[col * this->_rows + row];

	return this->_data[row * this->_cols + col];
}
template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
inline const T& Matrix<T, RowMajor, Container>::operator()(size_t index) const
{
	return this->_data[index];
}
template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
inline const T& Matrix<T, RowMajor, Container>::operator[](size_t index) const
{
	return this->_data[index];
}
template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
inline bool Matrix<T, RowMajor, Container>::operator==(const Matrix& other) const
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
template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
inline Matrix<T, RowMajor, Container> Matrix<T, RowMajor, Container>::operator+(const Matrix& other) const
{
	if (this->cols() != other.cols() || this->rows() != other.rows())
		throw std::invalid_argument("Matrix dimensions must agree for addition.");

	return this->add_copy(other);
}
template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
inline Matrix<T, RowMajor, Container> Matrix<T, RowMajor, Container>::operator-(const Matrix& other) const
{
	if (this->cols() != other.cols() || this->rows() != other.rows())
		throw std::invalid_argument("Matrix dimensions must agree for subtraction.");

	return this->sub_copy(other);
}
template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
inline Matrix<T, RowMajor, Container> Matrix<T, RowMajor, Container>::operator*(const Matrix& other) const
{
	if (this->cols() != other.rows())
		throw std::invalid_argument("Incompatible matrix dimensions for multiplication.");

	return this->matrix_mul_copy(other);
}
template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
inline Matrix<T, RowMajor, Container> Matrix<T, RowMajor, Container>::operator^(const Matrix& other) const
{
	if (this->cols() != other.cols() || this->rows() != other.rows())
		throw std::invalid_argument("Matrix dimensions must agree for Hadamard multiplication.");

	return this->hadamard_mul_copy(other);
}
template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
inline Matrix<T, RowMajor, Container> Matrix<T, RowMajor, Container>::operator/(const Matrix& other) const
{
	if (this->cols() != other.cols() || this->rows() != other.rows())
		throw std::invalid_argument("Matrix dimensions must agree for Hadamard division.");

	return this->hadamard_div_copy(other);
}
template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
inline bool Matrix<T, RowMajor, Container>::operator!=(const Matrix& other) const
{
	return !(*this == other);
}
template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
inline bool Matrix<T, RowMajor, Container>::operator==(const Matrix<T,!RowMajor>& other) const
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
template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
inline bool Matrix<T, RowMajor, Container>::operator!=(const Matrix<T,!RowMajor>& other) const
{
	return !(*this == other);
}
template<typename Ty, bool RowMajor, typename Container> requires VectorOrArray<Container>
std::ostream& operator<<(std::ostream& os, const Matrix<Ty, RowMajor, Container>& mat)
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