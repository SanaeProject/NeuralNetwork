#ifndef NEURALNETWORK_LAYER_H
#define NEURALNETWORK_LAYER_H

#include "Matrix.h"
#include "ActivationFunctions.h"

namespace NeuralNetwork {

/**
 * @brief Represents a layer in a neural network
 */
class Layer {
private:
    Matrix weights;
    Matrix bias;
    ActivationType activationType;
    
    // Cache for backpropagation
    Matrix lastInput;
    Matrix lastOutput;
    Matrix lastActivation;
    
public:
    // Constructor
    Layer(size_t inputSize, size_t outputSize, ActivationType activation = ActivationType::SIGMOID);
    
    // Forward pass
    Matrix forward(const Matrix& input);
    
    // Backward pass
    Matrix backward(const Matrix& outputGradient, double learningRate);
    
    // Getters
    const Matrix& getWeights() const { return weights; }
    const Matrix& getBias() const { return bias; }
    const Matrix& getLastInput() const { return lastInput; }
    const Matrix& getLastOutput() const { return lastOutput; }
    const Matrix& getLastActivation() const { return lastActivation; }
    ActivationType getActivationType() const { return activationType; }
    
    // Setters
    void setWeights(const Matrix& w) { weights = w; }
    void setBias(const Matrix& b) { bias = b; }
    
    // Initialize weights with different strategies
    void initializeWeightsRandom(double min = -1.0, double max = 1.0);
    void initializeWeightsXavier();
    void initializeWeightsHe();
};

} // namespace NeuralNetwork

#endif // NEURALNETWORK_LAYER_H
