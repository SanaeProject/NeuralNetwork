#ifndef SANAE_NEURALNETWORK_MATRIX  
#define SANAE_NEURALNETWORK_MATRIX  

#include <array>  
#include <initializer_list>  
#include <type_traits>  
#include <vector>  

// std::arrayを使う型
template<typename T>  
struct is_std_array : std::false_type {};  
template<typename T, std::size_t N>  
struct is_std_array<std::array<T, N>> : std::true_type {};  

// std::vectorを使う型
template<typename T>  
struct is_std_vector : std::false_type {};  
template<typename T, typename Alloc>  
struct is_std_vector<std::vector<T, Alloc>> : std::true_type {};  

// std::vectorまたはstd::array判定用の型
template<typename T>  
struct is_vector_or_array : std::disjunction<is_std_vector<T>, is_std_array<T>> {};  

template<typename T, bool RowMajor = true, typename Container = std::vector<T>,
typename En = std::enable_if_t<is_vector_or_array<Container>::value>>
class Matrix {
protected:
	size_t _rows, _cols;
	Container _data;

public:
	using Container2D = std::vector<std::vector<T>>;
	using InitContainer2D = std::initializer_list<std::initializer_list<T>>;

	Matrix();
	Matrix(size_t rows, size_t cols);
	Matrix(size_t rows, size_t cols, const T& initial);

	size_t rows() const;
	size_t cols() const;

	const Container& data() const;

	T& operator()(size_t row, size_t col);
	T& operator()(size_t index);
	
	const T& operator()(size_t row, size_t col) const;
	const T& operator()(size_t index) const;

	Matrix(const Container2D& data);
	Matrix(const InitContainer2D& data);
};

#endif // SANAE_NEURALNETWORK_MATRIX