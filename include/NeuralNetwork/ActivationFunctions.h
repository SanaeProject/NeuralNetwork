#ifndef NEURALNETWORK_ACTIVATIONFUNCTIONS_H
#define NEURALNETWORK_ACTIVATIONFUNCTIONS_H

#include "Matrix.h"
#include <cmath>

namespace NeuralNetwork {

/**
 * @brief Activation functions and their derivatives for neural networks
 */
namespace Activation {

// Sigmoid activation function
Matrix sigmoid(const Matrix& input);
Matrix sigmoidDerivative(const Matrix& output);

// ReLU (Rectified Linear Unit) activation function
Matrix relu(const Matrix& input);
Matrix reluDerivative(const Matrix& input);

// Tanh activation function
Matrix tanh(const Matrix& input);
Matrix tanhDerivative(const Matrix& output);

// Softmax activation function
Matrix softmax(const Matrix& input);

// Linear activation function (identity)
Matrix linear(const Matrix& input);
Matrix linearDerivative(const Matrix& input);

// Leaky ReLU activation function
Matrix leakyRelu(const Matrix& input, double alpha = 0.01);
Matrix leakyReluDerivative(const Matrix& input, double alpha = 0.01);

} // namespace Activation

/**
 * @brief Enum for activation function types
 */
enum class ActivationType {
    SIGMOID,
    RELU,
    TANH,
    SOFTMAX,
    LINEAR,
    LEAKY_RELU
};

} // namespace NeuralNetwork

#endif // NEURALNETWORK_ACTIVATIONFUNCTIONS_H
