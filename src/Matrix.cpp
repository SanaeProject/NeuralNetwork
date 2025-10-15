#include "NeuralNetwork/Matrix.h"
#include <cblas.h>
#include <algorithm>
#include <iomanip>

namespace NeuralNetwork {

// Constructors
Matrix::Matrix() : rows(0), cols(0) {}

Matrix::Matrix(size_t rows, size_t cols) 
    : data(rows * cols, 0.0), rows(rows), cols(cols) {}

Matrix::Matrix(size_t rows, size_t cols, double value)
    : data(rows * cols, value), rows(rows), cols(cols) {}

Matrix::Matrix(size_t rows, size_t cols, const std::vector<double>& data)
    : data(data), rows(rows), cols(cols) {
    if (data.size() != rows * cols) {
        throw std::invalid_argument("Data size must match rows * cols");
    }
}

Matrix::Matrix(const Matrix& other)
    : data(other.data), rows(other.rows), cols(other.cols) {}

Matrix::Matrix(Matrix&& other) noexcept
    : data(std::move(other.data)), rows(other.rows), cols(other.cols) {
    other.rows = 0;
    other.cols = 0;
}

// Assignment operators
Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        data = other.data;
        rows = other.rows;
        cols = other.cols;
    }
    return *this;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        data = std::move(other.data);
        rows = other.rows;
        cols = other.cols;
        other.rows = 0;
        other.cols = 0;
    }
    return *this;
}

// Element access
double& Matrix::operator()(size_t row, size_t col) {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix indices out of range");
    }
    return data[row * cols + col];
}

const double& Matrix::operator()(size_t row, size_t col) const {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix indices out of range");
    }
    return data[row * cols + col];
}

// Matrix operations
Matrix Matrix::operator+(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    
    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }
    
    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] - other.data[i];
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    
    Matrix result(rows, other.cols);
    
    // Use OpenBLAS for matrix multiplication
    // C = alpha * A * B + beta * C
    // where alpha = 1.0, beta = 0.0
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rows, other.cols, cols,
                1.0, data.data(), cols,
                other.data.data(), other.cols,
                0.0, result.data.data(), other.cols);
    
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] * scalar;
    }
    return result;
}

Matrix& Matrix::operator+=(const Matrix& other) {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] += other.data[i];
    }
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }
    
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] -= other.data[i];
    }
    return *this;
}

Matrix& Matrix::operator*=(double scalar) {
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] *= scalar;
    }
    return *this;
}

// Element-wise operations
Matrix Matrix::elementWiseMultiply(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for element-wise multiplication");
    }
    
    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] * other.data[i];
    }
    return result;
}

Matrix Matrix::elementWiseDivide(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for element-wise division");
    }
    
    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
        if (std::abs(other.data[i]) < 1e-10) {
            throw std::runtime_error("Division by zero");
        }
        result.data[i] = data[i] / other.data[i];
    }
    return result;
}

// Matrix transformations
Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.data[j * rows + i] = data[i * cols + j];
        }
    }
    return result;
}

// Utility functions
void Matrix::fill(double value) {
    std::fill(data.begin(), data.end(), value);
}

void Matrix::randomize(double min, double max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(min, max);
    
    for (auto& value : data) {
        value = dist(gen);
    }
}

void Matrix::apply(const std::function<double(double)>& func) {
    for (auto& value : data) {
        value = func(value);
    }
}

Matrix Matrix::map(const std::function<double(double)>& func) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = func(data[i]);
    }
    return result;
}

// Static factory methods
Matrix Matrix::zeros(size_t rows, size_t cols) {
    return Matrix(rows, cols, 0.0);
}

Matrix Matrix::ones(size_t rows, size_t cols) {
    return Matrix(rows, cols, 1.0);
}

Matrix Matrix::random(size_t rows, size_t cols, double min, double max) {
    Matrix result(rows, cols);
    result.randomize(min, max);
    return result;
}

Matrix Matrix::identity(size_t size) {
    Matrix result(size, size, 0.0);
    for (size_t i = 0; i < size; ++i) {
        result.data[i * size + i] = 1.0;
    }
    return result;
}

// Display
void Matrix::print() const {
    std::cout << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            std::cout << std::setw(10) << data[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

// Statistics
double Matrix::sum() const {
    double total = 0.0;
    for (const auto& value : data) {
        total += value;
    }
    return total;
}

double Matrix::mean() const {
    return sum() / static_cast<double>(data.size());
}

double Matrix::max() const {
    return *std::max_element(data.begin(), data.end());
}

double Matrix::min() const {
    return *std::min_element(data.begin(), data.end());
}

// Scalar multiplication (scalar on left side)
Matrix operator*(double scalar, const Matrix& matrix) {
    return matrix * scalar;
}

} // namespace NeuralNetwork
