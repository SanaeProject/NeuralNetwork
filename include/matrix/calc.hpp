#ifndef SANAE_NEURALNETWORK_MATRIX_CALC
#define SANAE_NEURALNETWORK_MATRIX_CALC

#include "matrix.h"
#include "view/view.h"
#include <functional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

template<typename T, bool RowMajor, typename Container, typename En>
template<typename execType, typename calcType, typename TyCheck>
inline void Matrix<T, RowMajor, Container, En>::_calc(Container& to, const Container& other, execType execPolicy, calcType operation)
{
	if (to.size() != other.size()) {
		throw std::invalid_argument("Container sizes must agree for calculation.");
	}

	std::transform(execPolicy,
		to.begin(), to.end(),
		other.begin(),
		to.begin(),
		operation);
}
template<typename T, bool RowMajor, typename Container, typename En>
template<typename execType, typename calcType, typename TyCheck>
inline void Matrix<T, RowMajor, Container, En>::_calc(Container& to, const T& other, execType execPolicy, calcType operation)
{
	std::transform(execPolicy,
		to.begin(), to.end(),
		to.begin(),
		[&](const T& val) { return operation(val, other); });
}
template<typename T, bool RowMajor, typename Container, typename En>
template<typename execType, typename TyCheck>
inline Matrix<T, RowMajor, Container, En>& Matrix<T, RowMajor, Container, En>::add(const Matrix& other, execType execPolicy)
{
	if (this->_rows != other._rows || this->_cols != other._cols) {
		throw std::invalid_argument("Matrix dimensions must agree for addition.");
	}

	_calc(this->_data, other._data, execPolicy, std::plus<T>());
	return *this;
}
template<typename T, bool RowMajor, typename Container, typename En>
template<typename execType, typename TyCheck>
inline Matrix<T, RowMajor, Container, En>& Matrix<T, RowMajor, Container, En>::sub(const Matrix& other, execType execPolicy)
{
	if (this->_rows != other._rows || this->_cols != other._cols) {
		throw std::invalid_argument("Matrix dimensions must agree for subtraction.");
	}

	_calc(this->_data, other._data, execPolicy, std::minus<T>());
	return *this;
}
template<typename T, bool RowMajor, typename Container, typename En>
template<typename execType, typename TyCheck>
inline Matrix<T, RowMajor, Container, En>& Matrix<T, RowMajor, Container, En>::hadamard_mul(const Matrix& other, execType execPolicy)
{
	if (this->_rows != other._rows || this->_cols != other._cols) {
		throw std::invalid_argument("Matrix dimensions must agree for Hadamard multiplication.");
	}

	_calc(this->_data, other._data, execPolicy, std::multiplies<T>());
	return *this;
}
template<typename T, bool RowMajor, typename Container, typename En>
template<typename execType, typename TyCheck>
inline Matrix<T, RowMajor, Container, En>& Matrix<T, RowMajor, Container, En>::scalar_mul(const T& scalar, execType execPolicy)
{
	_calc(this->_data, scalar, execPolicy, std::multiplies<T>());
	return *this;
}
template<typename T, bool RowMajor, typename Container, typename En>
template<typename execType, typename TyCheck>
inline Matrix<T, RowMajor, Container, En>& Matrix<T, RowMajor, Container, En>::hadamard_div(const Matrix<T, RowMajor, Container, En>& other, execType execPolicy)
{
	if (this->_rows != other._rows || this->_cols != other._cols) {
		throw std::invalid_argument("Matrix dimensions must agree for Hadamard division.");
	}

	_calc(this->_data, other._data, execPolicy, 
		[](const T& a, const T& b) {
			if (b == T(0)) {
				throw std::invalid_argument("Division by zero in Hadamard division.");
			}
			return a / b;
		}
	);
	return *this;
}
template<typename T, bool RowMajor, typename Container, typename En>
template<typename execType, typename TyCheck>
inline Matrix<T, RowMajor, Container, En>& Matrix<T, RowMajor, Container, En>::scalar_div(const T& scalar, execType execPolicy)
{
	if (scalar == T(0)) {
		throw std::invalid_argument("Division by zero in scalar_div.");
	}

	_calc(this->_data, scalar, execPolicy, std::divides<T>());
	return *this;
}
template<typename T, bool RowMajor, typename Container, typename En>
template<bool OtherMajor, typename MCheck>
inline Matrix<T, RowMajor, Container, En>& Matrix<T, RowMajor, Container, En>::matrix_mul(const Matrix<T, OtherMajor>& other)
{
	if (this->cols() != other.rows()) {
		throw std::invalid_argument("Matrix dimensions must agree for matrix multiplication.");
	}

	const size_t result_rows = this->rows();
	const size_t result_cols = other.cols();

	Container result_data(result_rows * result_cols);

	std::vector<View<T>> this_rows;
	std::vector<View<const T>> other_cols;

	this_rows.reserve(this->rows());
	other_cols.reserve(other.cols());

	for (size_t i = 0; i < this->rows(); i++)
		this_rows.emplace_back(this->get_row(i));
	for (size_t j = 0; j < other.cols(); j++)
		other_cols.emplace_back(other.get_col(j));

	auto task = [&](size_t row, size_t col) {
		T value = std::transform_reduce(
			this_rows[row].begin(), this_rows[row].end(),
			other_cols[col].begin(),
			T{},
			std::plus<T>(),
			std::multiplies<T>()
		);

		if constexpr (RowMajor)
			result_data[row * result_cols + col] = value;
		else
			result_data[col * result_rows + row] = value;
		};

	std::vector<std::thread> threads;
	const size_t max_threads = std::max(std::thread::hardware_concurrency(),1u);
	const size_t total_tasks = result_rows * result_cols;
	const size_t tasks_per_thread = (total_tasks + max_threads - 1) / max_threads;
	
	for (size_t t = 0; t < max_threads; t++) {
		threads.emplace_back([&, t]() {
			size_t start = t * tasks_per_thread;
			size_t end = std::min(start + tasks_per_thread, total_tasks);
			for (size_t index = start; index < end; index++) {
				size_t row, col;
				if constexpr (RowMajor) {
					row = index / result_cols;
					col = index % result_cols;
				}
				else {
					col = index / result_rows;
					row = index % result_rows;
				}

				task(row, col);
			}
			});
	}

	for (auto& thread : threads) {
		thread.join();
	}

	this->_rows = result_rows;
	this->_cols = result_cols;
	this->_data = std::move(result_data);
	
	return *this;
}

#endif