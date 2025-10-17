#ifndef SANAE_NEURALNETWORK_MATRIX  
#define SANAE_NEURALNETWORK_MATRIX  

#include <array>  
#include <execution>
#include <initializer_list>
#include <iosfwd>
#include <type_traits>  
#include <vector>  

// std::executionポリシー判定用の型
template<typename T> struct is_std_exec_policy : std::false_type {};
template<> struct is_std_exec_policy<std::execution::sequenced_policy>            : std::true_type {};
template<> struct is_std_exec_policy<std::execution::parallel_policy>             : std::true_type {};
template<> struct is_std_exec_policy<std::execution::parallel_unsequenced_policy> : std::true_type {};

// std::vectorまたはstd::array判定用の型
template<typename T> struct is_vector_or_array : std::false_type {};
template<typename T, typename Alloc> struct is_vector_or_array<std::vector<T, Alloc>> : std::true_type {};
template<typename T, std::size_t N>  struct is_vector_or_array<std::array<T, N>>      : std::true_type {};

template<typename T, bool RowMajor = true, typename Container = std::vector<T>,
typename En = std::enable_if_t<is_vector_or_array<Container>::value>>
class Matrix {
protected:
	size_t _rows, _cols;
	Container _data;

	template<typename execType, typename calcType, typename TyCheck = std::enable_if_t<is_std_exec_policy<execType>::value>>
	void _calc(Container& to,const Container& other, execType execPolicy, calcType operation);
	template<typename execType, typename calcType, typename TyCheck = std::enable_if_t<is_std_exec_policy<execType>::value>>
	void _calc(Container& to, const T& other, execType execPolicy, calcType operation);
public:
	using Container2D = std::vector<std::vector<T>>;
	using InitContainer2D = std::initializer_list<std::initializer_list<T>>;

	// ctor.hpp
	Matrix();
	Matrix(size_t rows, size_t cols);
	Matrix(size_t rows, size_t cols, const T& initial);
	Matrix(const Container2D& data);
	Matrix(const InitContainer2D& data);
	Matrix(const Matrix& other)     = default;
	Matrix(Matrix&& other) noexcept = default;
	~Matrix() = default;

	// util.hpp
	size_t rows() const;
	size_t cols() const;

	const Container& data() const;
	Matrix<T, !RowMajor> convertLayout() const;

	// ops.hpp
	T& operator()(size_t row, size_t col);
	T& operator()(size_t index);
	
	const T& operator()(size_t row, size_t col) const;
	const T& operator()(size_t index) const;

	bool operator==(const Matrix& other) const;
	bool operator!=(const Matrix& other) const;

	Matrix& operator=(const Matrix& other) = default;

	bool operator==(const Matrix<T,!RowMajor>& other) const;
	bool operator!=(const Matrix<T,!RowMajor>& other) const;

	template<typename T, bool R, typename C, typename E>
	friend std::ostream& operator<<(std::ostream& os, const Matrix<T,R,C,E>& mat);

	// calc.hpp
	template<typename execType = std::execution::sequenced_policy, typename TyCheck = std::enable_if_t<is_std_exec_policy<execType>::value>>
	Matrix& add(const Matrix& other,execType execPolicy=execType());
	template<typename execType = std::execution::sequenced_policy, typename TyCheck = std::enable_if_t<is_std_exec_policy<execType>::value>>
	Matrix& sub(const Matrix& other, execType execPolicy=execType());
	template<typename execType = std::execution::sequenced_policy, typename TyCheck = std::enable_if_t<is_std_exec_policy<execType>::value>>
	Matrix& hadamard_mul(const Matrix& other, execType execPolicy = execType());	
	template<typename execType = std::execution::sequenced_policy, typename TyCheck = std::enable_if_t<is_std_exec_policy<execType>::value>>
	Matrix& scalar_mul(const T& scalar, execType execPolicy=execType());
	
	template<typename execType = std::execution::sequenced_policy, typename TyCheck = std::enable_if_t<is_std_exec_policy<execType>::value>>
	Matrix& hadamard_div(const Matrix& other, execType execPolicy = execType());
	template<typename execType = std::execution::sequenced_policy, typename TyCheck = std::enable_if_t<is_std_exec_policy<execType>::value>>
	Matrix& scalar_div(const T& scalar, execType execPolicy = execType());

	template<typename execType = std::execution::sequenced_policy, typename TyCheck = std::enable_if_t<is_std_exec_policy<execType>::value>>
	Matrix& matrix_mul(const Matrix& other, execType execPolicy = execType());
};

#endif // SANAE_NEURALNETWORK_MATRIX