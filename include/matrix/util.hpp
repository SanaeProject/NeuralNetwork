#ifndef SANAE_NEURALNETWORK_MATRIX_UTIL
#define SANAE_NEURALNETWORK_MATRIX_UTIL

#include "../view/view.h"
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
	
	const size_t rows = this->rows();
	const size_t cols = this->cols();
	for (size_t i = 0; i < rows; i++) {
		size_t base;

		if constexpr (RowMajor)
			base = i * cols;  // s—Dæ‚©‚ç—ñ—Dæ‚Ö•ÏŠ·
		else
			base = i * rows;  // —ñ—Dæ‚©‚çs—Dæ‚Ö•ÏŠ·

		for (size_t j = 0; j < cols; j++) {
			if constexpr (RowMajor) {
				result[j * cols + i] = this->_data[base + j]; // result[]
			}
			else {
				result[base + j] = this->_data[j * rows + i];
			}
		}
	}

	return result;
}
template<typename T, bool RowMajor, typename Container, typename En>
inline View<T> Matrix<T, RowMajor, Container, En>::get_row(size_t row)
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
template<typename T, bool RowMajor, typename Container, typename En>
inline View<T> Matrix<T, RowMajor, Container, En>::get_col(size_t col)
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
template<typename T, bool RowMajor, typename Container, typename En>
inline View<const T> Matrix<T, RowMajor, Container, En>::get_row(size_t row) const
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
template<typename T, bool RowMajor, typename Container, typename En>
inline View<const T> Matrix<T, RowMajor, Container, En>::get_col(size_t col) const
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

#endif // SANAE_NEURALNETWORK_MATRIX_UTIL