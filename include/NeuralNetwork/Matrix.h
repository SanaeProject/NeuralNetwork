#ifndef NEURALNETWORK_MATRIX_H
#define NEURALNETWORK_MATRIX_H

#include <vector>
#include <iostream>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <random>

namespace NeuralNetwork {

/**
 * @brief Matrix class using OpenBLAS for efficient linear algebra operations
 */
class Matrix {
private:
    std::vector<double> data;
    size_t rows;
    size_t cols;

public:
    // Constructors
    Matrix();
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, double value);
    Matrix(size_t rows, size_t cols, const std::vector<double>& data);
    Matrix(const Matrix& other);
    Matrix(Matrix&& other) noexcept;
    
    // Assignment operators
    Matrix& operator=(const Matrix& other);
    Matrix& operator=(Matrix&& other) noexcept;
    
    // Destructor
    ~Matrix() = default;

    // Getters
    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }
    size_t size() const { return rows * cols; }
    const std::vector<double>& getData() const { return data; }
    
    // Element access
    double& operator()(size_t row, size_t col);
    const double& operator()(size_t row, size_t col) const;
    
    // Matrix operations
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const; // Matrix multiplication
    Matrix operator*(double scalar) const;
    
    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix& operator*=(double scalar);
    
    // Element-wise operations
    Matrix elementWiseMultiply(const Matrix& other) const;
    Matrix elementWiseDivide(const Matrix& other) const;
    
    // Matrix transformations
    Matrix transpose() const;
    
    // Utility functions
    void fill(double value);
    void randomize(double min = -1.0, double max = 1.0);
    void apply(const std::function<double(double)>& func);
    Matrix map(const std::function<double(double)>& func) const;
    
    // Static factory methods
    static Matrix zeros(size_t rows, size_t cols);
    static Matrix ones(size_t rows, size_t cols);
    static Matrix random(size_t rows, size_t cols, double min = -1.0, double max = 1.0);
    static Matrix identity(size_t size);
    
    // Display
    void print() const;
    
    // Statistics
    double sum() const;
    double mean() const;
    double max() const;
    double min() const;
};

// Scalar multiplication (scalar on left side)
Matrix operator*(double scalar, const Matrix& matrix);

} // namespace NeuralNetwork

#endif // NEURALNETWORK_MATRIX_H
