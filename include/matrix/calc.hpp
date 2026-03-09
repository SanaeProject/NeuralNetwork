#ifndef SANAE_NEURALNETWORK_MATRIX_CALC
#define SANAE_NEURALNETWORK_MATRIX_CALC

#include "../view/view.h"
#include "blasgemm.h"
#include "matrix.h"
#include <functional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>
#include <algorithm>

template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
template<typename execType, typename calcType>
inline void Matrix<T, RowMajor, Container>::_calc(Container& to, const Container& other, execType execPolicy, calcType operation)
	requires StdExecPolicy<execType>
{
    if (to.size() != other.size())
        throw std::invalid_argument("Container sizes must agree for calculation.");

    std::transform(execPolicy,
        to.begin(), to.end(),
        other.begin(),
        to.begin(),
        operation);
}

template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
template<typename execType, typename calcType>
inline void Matrix<T, RowMajor, Container>::_calc(Container& to, const T& other, execType execPolicy, calcType operation)
	requires StdExecPolicy<execType>
{
    std::transform(execPolicy,
        to.begin(), to.end(),
        to.begin(),
        [&](const T& val) { return operation(val, other); });
}
template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
template<bool use_blas, typename execType>
inline Matrix<T, RowMajor, Container>& Matrix<T, RowMajor, Container>::add(const Matrix& other, execType execPolicy) requires StdExecPolicy<execType>
{
	if (this->_rows != other._rows || this->_cols != other._cols)
		throw std::invalid_argument("Matrix dimensions must agree for addition.");

	if constexpr (can_use_blas<T>::value && use_blas) {
		int n = static_cast<int>(this->_rows * this->_cols);
		BlasGemm::Add<T>::axpy(n, 1.0, other._data.data(), this->_data.data());
	}
	else {
		this->_calc(this->_data, other._data, execPolicy, std::plus<T>());
	}
	return *this;
}
template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
template<bool use_blas, typename execType>
inline Matrix<T, RowMajor, Container>& Matrix<T, RowMajor, Container>::sub(const Matrix& other, execType execPolicy) requires StdExecPolicy<execType>
{
	if (this->_rows != other._rows || this->_cols != other._cols)
		throw std::invalid_argument("Matrix dimensions must agree for subtraction.");

	if constexpr (can_use_blas<T>::value && use_blas) {
		int n = static_cast<int>(this->_rows * this->_cols);
		BlasGemm::Sub<T>::axpy(n, 1.0, other._data.data(), this->_data.data());
	}
	else {
		this->_calc(this->_data, other._data, execPolicy, std::minus<T>());
	}

	return *this;
}
template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
template<typename execType>
inline Matrix<T, RowMajor, Container>& Matrix<T, RowMajor, Container>::hadamard_mul(const Matrix& other, execType execPolicy) requires StdExecPolicy<execType>
{
   if (this->_rows != other._rows || this->_cols != other._cols)
       throw std::invalid_argument("Matrix dimensions must agree for Hadamard multiplication.");

   this->_calc(this->_data, other._data, execPolicy, std::multiplies<T>());
   return *this;
}
template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
template<typename execType>
inline Matrix<T, RowMajor, Container>& Matrix<T, RowMajor, Container>::hadamard_div(const Matrix<T, RowMajor, Container>& other, execType execPolicy) requires StdExecPolicy<execType>
{
	if (this->_rows != other._rows || this->_cols != other._cols)
		throw std::invalid_argument("Matrix dimensions must agree for Hadamard division.");

	this->_calc(this->_data, other._data, execPolicy, 
		[](const T& a, const T& b) {
			if (b == T(0)) {
				throw std::invalid_argument("Division by zero in Hadamard division.");
			}
			return a / b;
		}
	);
	return *this;
}
template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
template<bool use_blas, typename execType>
inline Matrix<T, RowMajor, Container>& Matrix<T, RowMajor, Container>::scalar_mul(const T& scalar, execType execPolicy) requires StdExecPolicy<execType>
{
	if constexpr (can_use_blas<T>::value && use_blas) {
		int n = static_cast<int>(this->_rows * this->_cols);
		BlasGemm::ScalarMul<T>::scal(n, scalar, this->_data.data());
	}
	else {
		this->_calc(this->_data, scalar, execPolicy, std::multiplies<T>());
	}
	return *this;
}
template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
template<typename execType>
inline Matrix<T, RowMajor, Container>& Matrix<T, RowMajor, Container>::scalar_div(const T& scalar, execType execPolicy) requires StdExecPolicy<execType>
{
	if (scalar == T(0))
		throw std::invalid_argument("Division by zero in scalar_div.");
	
	this->_calc(this->_data, scalar, execPolicy, std::divides<T>());
	return *this;
}
template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
template<bool use_blas, bool OtherMajor, typename MCheck>
inline Matrix<T, RowMajor, Container>& Matrix<T, RowMajor, Container>::matrix_mul(const Matrix<T, OtherMajor, Container>& other)
{
	if (this->cols() != other.rows()) {
		throw std::invalid_argument("Matrix dimensions must agree for matrix multiplication.");
	}

	const size_t result_rows = this->rows();
	const size_t result_cols = other.cols();

	Container result_data;
	if constexpr (is_std_array<Container>::value) {
		result_data = Container();
	}else
	{
		result_data = Container(result_rows * result_cols);
	}

	if constexpr (can_use_blas<T>::value && use_blas) {
		int m = static_cast<int>(result_rows);
		int n = static_cast<int>(result_cols);
		int k = static_cast<int>(this->cols());

		BlasGemm::MatMul<T>::multiply(
			this->_data.data(),
			other.data().data(),
			result_data.data(),
			m, n, k,
			RowMajor,
			OtherMajor
		);
	}else{
		std::vector<View<T>> this_rows;
		std::vector<View<const T>> other_cols;

		this_rows.reserve(result_rows);
		other_cols.reserve(result_cols);

		for (size_t i = 0; i < result_rows; i++)
			this_rows.emplace_back(this->get_row(i));
		for (size_t j = 0; j < result_cols; j++)
			other_cols.emplace_back(other.get_col(j));

		auto task = [&](size_t start, size_t end) {
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
			}
			};

		std::vector<std::thread> threads;
		const size_t max_threads = std::max(std::thread::hardware_concurrency(), 1u);
		const size_t total_tasks = result_rows * result_cols;
		const size_t tasks_per_thread = (total_tasks + max_threads - 1) / max_threads;

		for (size_t t = 0; t < max_threads; t++) {
			threads.emplace_back([&, t]() {
				size_t start = t * tasks_per_thread;
				size_t end = std::min(start + tasks_per_thread, total_tasks);
				task(start, end);
				});
		}

		for (auto& thread : threads) {
			thread.join();
		}
	}

	this->_rows = result_rows;
	this->_cols = result_cols;
	this->_data = std::move(result_data);
	
	return *this;
}

#endif