#ifndef SANAE_NEURALNETWORK_CONSEPTS_HPP
#define SANAE_NEURALNETWORK_CONSEPTS_HPP

#include <type_traits>
#include <vector>
#include <array>

// std::executionポリシー判定用の型
template<typename T>
concept StdExecPolicy = std::is_execution_policy_v<std::remove_cvref_t<T>>;

// std::vectorまたはstd::array判定用の型
template<typename T> struct is_vector_or_array : std::false_type {};
template<typename T, typename Alloc> struct is_vector_or_array<std::vector<T, Alloc>> : std::true_type {};
template<typename T, std::size_t N>  struct is_vector_or_array<std::array<T, N>>      : std::true_type {};
template<typename T> 
concept VectorOrArray = is_vector_or_array<std::remove_cvref_t<T>>::value;

// std::array判定用の型
template<typename T> struct is_std_array : std::false_type {};
template<typename T, std::size_t N>  struct is_std_array<std::array<T, N>> : std::true_type {};
template<typename T> 
concept StdArray = is_std_array<std::remove_cvref_t<T>>::value;

// BLAS使用判定用の型
template<typename T> struct can_use_blas : std::false_type {};
#if defined(USE_OPENBLAS)
// OpenBlas
	template<> struct can_use_blas<float>  : std::true_type {};
	template<> struct can_use_blas<double> : std::true_type {};
#elif defined(USE_CUBLAS)
// cuBLAS
	template<> struct can_use_blas<float> : std::true_type {};
	template<> struct can_use_blas<double> : std::true_type {};
#elif defined(USE_CLBLAST)
// clBLAST
	template<> struct can_use_blas<float> : std::true_type {};
	template<> struct can_use_blas<double> : std::true_type {};
#endif
template<typename T> concept CanUseBlas = can_use_blas<T>::value;

#endif // SANAE_NEURALNETWORK_CONSEPTS_HPP