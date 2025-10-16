#ifndef SANAE_NEURALNETWORK_MATRIX_CTOR  
#define SANAE_NEURALNETWORK_MATRIX_CTOR  

#include <type_traits>  
#include "matrix.h"  

template<typename T, bool RowMajor, typename Container, typename En>
inline Matrix<T, RowMajor, Container, En>::Matrix()
{  
}  
template<typename T, bool RowMajor, typename Container, typename En>
inline Matrix<T, RowMajor, Container, En>::Matrix(size_t rows, size_t cols) : _rows(rows), _cols(cols), _data(rows * cols)
{
}  
template<typename T, bool RowMajor, typename Container, typename En>
inline Matrix<T, RowMajor, Container, En>::Matrix(size_t rows, size_t cols, const T& initial) : _rows(rows), _cols(cols), _data(rows * cols, initial)
{  
}  
template<typename T, bool RowMajor, typename Container, typename En>
inline Matrix<T, RowMajor, Container, En>::Matrix(const Container2D& data)
{  
   this->_rows = data.size();  
   this->_cols = data.empty() ? 0 : data[0].size();  
   this->_data.resize(this->_rows * this->_cols);  

   const size_t outer = RowMajor ? _rows : _cols;  
   const size_t inner = RowMajor ? _cols : _rows;

   size_t index = 0;
   for (size_t i = 0; i < outer; i++) {
       for (size_t j = 0; j < inner; j++) {  
           _data[index] = RowMajor ? data[i][j] : data[j][i];
           index++;
       }  
   }  
}  
template<typename T, bool RowMajor, typename Container, typename En>
inline Matrix<T, RowMajor, Container, En>::Matrix(const InitContainer2D& data)
{  
   this->_rows = data.size();  
   this->_cols = data.begin()->size();  
   this->_data.resize(this->_rows * this->_cols);  

   const size_t outer = RowMajor ? _rows : _cols;  
   const size_t inner = RowMajor ? _cols : _rows;  
   size_t index = 0;
   for (size_t i = 0; i < outer; i++) {
       for (size_t j = 0; j < inner; j++) {
           _data[index] = RowMajor ? *((data.begin()+i)->begin()+j) : *((data.begin() + j)->begin() + i);
           index++;
       }
   }
}  

#endif // SANAE_NEURALNETWORK_MATRIX_CTOR