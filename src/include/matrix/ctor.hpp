#ifndef SANAE_NEURALNETWORK_MATRIX_CTOR  
#define SANAE_NEURALNETWORK_MATRIX_CTOR  

#include "matrix.h" 
#include <algorithm>

template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
inline Matrix<T, RowMajor, Container>::Matrix() : _rows(0), _cols(0), _data()
{  
}
template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
inline Matrix<T, RowMajor, Container>::Matrix(size_t rows, size_t cols)
{
    this->_rows = rows;
    this->_cols = cols;

    if constexpr (is_std_array<Container>::value) {
        // std::arrayのサイズチェックを追加
        if (this->_rows * this->_cols > std::tuple_size_v<Container>) {
            throw std::invalid_argument("Matrix dimensions do not match std::array size");
        }
        this->_data = Container();
    }
    else {
        this->_data = Container(this->_rows * this->_cols);
    }
}
template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
template<typename InitFunc, typename ExecPolicy>
inline Matrix<T, RowMajor, Container>::Matrix(size_t rows, size_t cols, InitFunc func, ExecPolicy execPolicy) 
requires
    std::convertible_to<std::invoke_result_t<InitFunc>, T> &&
    StdExecPolicy<ExecPolicy>
{
    this->_rows = rows;
    this->_cols = cols;

    if constexpr (is_std_array<Container>::value) {
        // std::arrayのサイズチェックを追加
        if (this->_rows * this->_cols > std::tuple_size_v<Container>) {
            throw std::invalid_argument("Matrix dimensions do not match std::array size");
        }
        this->_data = Container();
    }
    else {
        this->_data = Container(this->_rows * this->_cols);
    }

    std::generate(execPolicy, this->_data.begin(), this->_data.end(), func);
}
template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
inline Matrix<T, RowMajor, Container>::Matrix(const Container2D& data)
{  
   this->_rows = data.size();  
   this->_cols = data.empty() ? 0 : data[0].size();

   if constexpr (is_std_array<Container>::value) {
       // std::arrayのサイズチェックを追加
       if (this->_rows * this->_cols > std::tuple_size_v<Container>) {
           throw std::invalid_argument("Matrix dimensions do not match std::array size");
       }
       this->_data = Container();
   }
   else {
       this->_data = Container(this->_rows * this->_cols);
   }

   const size_t outer = RowMajor ? _rows : _cols;  
   const size_t inner = RowMajor ? _cols : _rows;

   size_t index = 0;
   for (size_t i = 0; i < outer; i++) {
       for (size_t j = 0; j < inner; j++) {
           if constexpr (RowMajor)
               _data[index] = data[i][j];
           else
               _data[index] = data[j][i];

           index++;
       }  
   }  
}
template<typename T, bool RowMajor, typename Container> requires VectorOrArray<Container>
inline Matrix<T, RowMajor, Container>::Matrix(const InitContainer2D& data)
{  
   this->_rows = data.size();  
   this->_cols = data.size() == 0 ? 0 : data.begin()->size();
        
   if constexpr (is_std_array<Container>::value) {
       // std::arrayのサイズチェックを追加
       if (this->_rows * this->_cols > std::tuple_size_v<Container>) {
           throw std::invalid_argument("Matrix dimensions do not match std::array size");
       }
       this->_data = Container();
   }
   else {
       this->_data = Container(this->_rows * this->_cols);
   }

   const size_t outer = RowMajor ? _rows : _cols;  
   const size_t inner = RowMajor ? _cols : _rows;  
   size_t index = 0;
   for (size_t i = 0; i < outer; i++) {
       for (size_t j = 0; j < inner; j++) {
           if constexpr (RowMajor)
               _data[index] = *((data.begin() + i)->begin() + j);
           else
               _data[index] = *((data.begin() + j)->begin() + i);

           index++;
       }
   }
}

#endif // SANAE_NEURALNETWORK_MATRIX_CTOR