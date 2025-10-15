#include "NeuralNetwork/ActivationFunctions.h"
#include <algorithm>

namespace NeuralNetwork {
namespace Activation {

// Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
Matrix sigmoid(const Matrix& input) {
    return input.map([](double x) {
        return 1.0 / (1.0 + std::exp(-x));
    });
}

// Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
// where input is the output of sigmoid
Matrix sigmoidDerivative(const Matrix& output) {
    return output.map([](double y) {
        return y * (1.0 - y);
    });
}

// ReLU activation function: f(x) = max(0, x)
Matrix relu(const Matrix& input) {
    return input.map([](double x) {
        return std::max(0.0, x);
    });
}

// Derivative of ReLU: f'(x) = 1 if x > 0, else 0
Matrix reluDerivative(const Matrix& input) {
    return input.map([](double x) {
        return x > 0.0 ? 1.0 : 0.0;
    });
}

// Tanh activation function: f(x) = tanh(x)
Matrix tanh(const Matrix& input) {
    return input.map([](double x) {
        return std::tanh(x);
    });
}

// Derivative of tanh: f'(x) = 1 - tanh(x)^2
// where input is the output of tanh
Matrix tanhDerivative(const Matrix& output) {
    return output.map([](double y) {
        return 1.0 - y * y;
    });
}

// Softmax activation function
Matrix softmax(const Matrix& input) {
    Matrix result(input.getRows(), input.getCols());
    
    for (size_t i = 0; i < input.getRows(); ++i) {
        // Find max for numerical stability
        double maxVal = input(i, 0);
        for (size_t j = 1; j < input.getCols(); ++j) {
            maxVal = std::max(maxVal, input(i, j));
        }
        
        // Compute exp(x - max) and sum
        double sum = 0.0;
        for (size_t j = 0; j < input.getCols(); ++j) {
            result(i, j) = std::exp(input(i, j) - maxVal);
            sum += result(i, j);
        }
        
        // Normalize
        for (size_t j = 0; j < input.getCols(); ++j) {
            result(i, j) /= sum;
        }
    }
    
    return result;
}

// Linear activation function: f(x) = x
Matrix linear(const Matrix& input) {
    return input;
}

// Derivative of linear: f'(x) = 1
Matrix linearDerivative(const Matrix& input) {
    return Matrix::ones(input.getRows(), input.getCols());
}

// Leaky ReLU activation function: f(x) = x if x > 0, else alpha * x
Matrix leakyRelu(const Matrix& input, double alpha) {
    return input.map([alpha](double x) {
        return x > 0.0 ? x : alpha * x;
    });
}

// Derivative of Leaky ReLU: f'(x) = 1 if x > 0, else alpha
Matrix leakyReluDerivative(const Matrix& input, double alpha) {
    return input.map([alpha](double x) {
        return x > 0.0 ? 1.0 : alpha;
    });
}

} // namespace Activation
} // namespace NeuralNetwork
