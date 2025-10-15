#include "NeuralNetwork/Layer.h"
#include <cmath>

namespace NeuralNetwork {

Layer::Layer(size_t inputSize, size_t outputSize, ActivationType activation)
    : weights(outputSize, inputSize),
      bias(outputSize, 1),
      activationType(activation) {
    
    // Initialize with Xavier initialization by default
    initializeWeightsXavier();
    bias.fill(0.0);
}

Matrix Layer::forward(const Matrix& input) {
    // Store input for backpropagation
    lastInput = input;
    
    // Compute weighted sum: output = weights * input + bias
    lastOutput = weights * input;
    
    // Add bias to each column
    for (size_t col = 0; col < lastOutput.getCols(); ++col) {
        for (size_t row = 0; row < lastOutput.getRows(); ++row) {
            lastOutput(row, col) += bias(row, 0);
        }
    }
    
    // Apply activation function
    switch (activationType) {
        case ActivationType::SIGMOID:
            lastActivation = Activation::sigmoid(lastOutput);
            break;
        case ActivationType::RELU:
            lastActivation = Activation::relu(lastOutput);
            break;
        case ActivationType::TANH:
            lastActivation = Activation::tanh(lastOutput);
            break;
        case ActivationType::SOFTMAX:
            lastActivation = Activation::softmax(lastOutput);
            break;
        case ActivationType::LINEAR:
            lastActivation = Activation::linear(lastOutput);
            break;
        case ActivationType::LEAKY_RELU:
            lastActivation = Activation::leakyRelu(lastOutput);
            break;
    }
    
    return lastActivation;
}

Matrix Layer::backward(const Matrix& outputGradient, double learningRate) {
    // Compute gradient with respect to activation
    Matrix activationGradient;
    
    switch (activationType) {
        case ActivationType::SIGMOID:
            activationGradient = outputGradient.elementWiseMultiply(
                Activation::sigmoidDerivative(lastActivation));
            break;
        case ActivationType::RELU:
            activationGradient = outputGradient.elementWiseMultiply(
                Activation::reluDerivative(lastOutput));
            break;
        case ActivationType::TANH:
            activationGradient = outputGradient.elementWiseMultiply(
                Activation::tanhDerivative(lastActivation));
            break;
        case ActivationType::SOFTMAX:
            // For softmax, we assume cross-entropy loss is used
            // and the gradient is simplified
            activationGradient = outputGradient;
            break;
        case ActivationType::LINEAR:
            activationGradient = outputGradient.elementWiseMultiply(
                Activation::linearDerivative(lastOutput));
            break;
        case ActivationType::LEAKY_RELU:
            activationGradient = outputGradient.elementWiseMultiply(
                Activation::leakyReluDerivative(lastOutput));
            break;
    }
    
    // Compute gradients
    Matrix inputTranspose = lastInput.transpose();
    Matrix weightsGradient = activationGradient * inputTranspose;
    
    // Compute bias gradient (sum over batch)
    Matrix biasGradient(bias.getRows(), 1, 0.0);
    for (size_t row = 0; row < activationGradient.getRows(); ++row) {
        double sum = 0.0;
        for (size_t col = 0; col < activationGradient.getCols(); ++col) {
            sum += activationGradient(row, col);
        }
        biasGradient(row, 0) = sum;
    }
    
    // Update weights and bias
    weights -= weightsGradient * learningRate;
    bias -= biasGradient * learningRate;
    
    // Compute gradient with respect to input
    Matrix weightsTranspose = weights.transpose();
    Matrix inputGradient = weightsTranspose * activationGradient;
    
    return inputGradient;
}

void Layer::initializeWeightsRandom(double min, double max) {
    weights.randomize(min, max);
}

void Layer::initializeWeightsXavier() {
    // Xavier initialization: std = sqrt(2 / (fan_in + fan_out))
    double fanIn = static_cast<double>(weights.getCols());
    double fanOut = static_cast<double>(weights.getRows());
    double std = std::sqrt(2.0 / (fanIn + fanOut));
    weights.randomize(-std, std);
}

void Layer::initializeWeightsHe() {
    // He initialization: std = sqrt(2 / fan_in)
    double fanIn = static_cast<double>(weights.getCols());
    double std = std::sqrt(2.0 / fanIn);
    weights.randomize(-std, std);
}

} // namespace NeuralNetwork
