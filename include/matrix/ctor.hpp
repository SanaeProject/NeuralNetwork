#ifndef SANAE_NEURALNETWORK_MATRIX_CTOR  
#define SANAE_NEURALNETWORK_MATRIX_CTOR  

#include "matrix.h"  
#include <initializer_list>

template<typename T, bool RowMajor, typename Container, typename En>
inline Matrix<T, RowMajor, Container, En>::Matrix() : _rows(0), _cols(0), _data()
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
template<typename InitFunc, typename InitFuncCheck>
inline Matrix<T, RowMajor, Container, En>::Matrix(size_t rows, size_t cols, InitFunc func)
{
    this->_rows = rows;
    this->_cols = cols;
    this->_data.resize(rows * cols);

    std::generate(this->_data.begin(), this->_data.end(), func);
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
           if constexpr (RowMajor)
			   _data[index] = data[i][j];
           else
			   _data[index] = data[j][i];

           index++;
       }  
   }  
}
template<typename T, bool RowMajor, typename Container, typename En>
inline Matrix<T, RowMajor, Container, En>::Matrix(const InitContainer2D& data)
{  
   this->_rows = data.size();  
   this->_cols = data.size() == 0 ? 0 : data.begin()->size();
   this->_data.resize(this->_rows * this->_cols);  

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