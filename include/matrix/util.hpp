#ifndef SANAE_NEURALNETWORK_MATRIX_UTIL
#define SANAE_NEURALNETWORK_MATRIX_UTIL

#include "../view/view.h"
#include "matrix.h"

template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
inline size_t Matrix<T, RowMajor, Container>::rows() const
{
	return this->_rows;
}
template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
inline size_t Matrix<T, RowMajor, Container>::cols() const
{
	return this->_cols;
}
template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
inline const Container& Matrix<T, RowMajor, Container>::data() const noexcept
{
	return this->_data;
}
template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
inline Matrix<T, !RowMajor> Matrix<T, RowMajor, Container>::convertLayout() const
{
	Matrix<T, !RowMajor> result(this->rows(), this->cols());
	
	const size_t rows = this->rows();
	const size_t cols = this->cols();
	for (size_t i = 0; i < rows; i++) {
		size_t base;

		if constexpr (RowMajor)
			base = i * cols;  // s—Dæ‚©‚ç—ñ—Dæ‚Ö•ÏŠ·
		else
			base = i * rows;  // —ñ—Dæ‚©‚çs—Dæ‚Ö•ÏŠ·

		// before[i,j] -> after[j,i] 
		for (size_t j = 0; j < cols; j++) {
			if constexpr (RowMajor) {
				result[j * cols + i] = this->_data[base + j]; // after[j*cols + i] = before[i*cols + j] 
			}
			else {
				result[base + j] = this->_data[j * rows + i]; // after[i*rows + j] = before[j*rows + i]
			}
		}
	}

	return result;
}
template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
inline View<T> Matrix<T, RowMajor, Container>::get_row(size_t row)
{
	if constexpr (RowMajor) {
		View<T> view(&this->_data[row * this->cols()], this->cols());
		return view;
	}
	else
	{
		View<T> view(&this->_data[row], this->cols(), this->rows());
		return view;
	}
}
template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
inline View<T> Matrix<T, RowMajor, Container>::get_col(size_t col)
{
	if constexpr (!RowMajor) {
		View<T> view(&this->_data[col * this->rows()], this->rows());
		return view;
	}
	else
	{
		View<T> view(&this->_data[col], this->rows(), this->cols());
		return view;
	}
}
template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
inline View<const T> Matrix<T, RowMajor, Container>::get_row(size_t row) const
{
	if constexpr (RowMajor) {
		View<const T> view(&this->_data[row * this->cols()], this->cols());
		return view;
	}
	else
	{
		View<const T> view(&this->_data[row], this->cols(), this->rows());
		return view;
	}
}
template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
inline View<const T> Matrix<T, RowMajor, Container>::get_col(size_t col) const
{
	if constexpr (!RowMajor) {
		View<const T> view(&this->_data[col * this->rows()], this->rows());
		return view;
	}
	else
	{
		View<const T> view(&this->_data[col], this->rows(), this->cols());
		return view;
	}
}
template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
inline bool Matrix<T, RowMajor, Container>::is_blas_enabled() const
{
	return can_use_blas<T>::value;
}

template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
inline Matrix<T, RowMajor, Container>& Matrix<T, RowMajor, Container>::transpose()
{
	Container result(this->_data.size());
	const size_t rows = this->rows();
	const size_t cols = this->cols();

	for (size_t i = 0; i < rows; i++) {
		size_t base;

		if constexpr (RowMajor)
			base = i * cols; // s—Dæ‚©‚ç—ñ—Dæ‚Ö•ÏŠ·
		else
			base = i * rows; // —ñ—Dæ‚©‚çs—Dæ‚Ö•ÏŠ·

		for (size_t j = 0; j < cols; j++) {
			if constexpr (RowMajor) {
				result[j * cols + i] = this->_data[base + j]; // after[j*cols + i] = before[i*cols + j]
			}
			else {
				result[base + j] = this->_data[j * rows + i]; // after[i*rows + j] = before[j*rows + i]
			}
		}
	}

	this->_rows = cols;
	this->_cols = rows;
	this->_data = std::move(result);
	return *this;
}

#endif // SANAE_NEURALNETWORK_MATRIX_UTIL