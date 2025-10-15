#ifndef NEURALNETWORK_NEURALNETWORK_H
#define NEURALNETWORK_NEURALNETWORK_H

#include "Matrix.h"
#include "Layer.h"
#include <vector>
#include <memory>

namespace NeuralNetwork {

/**
 * @brief Neural Network class for managing layers and training
 */
class NeuralNetworkModel {
private:
    std::vector<std::unique_ptr<Layer>> layers;
    double learningRate;
    
public:
    // Constructor
    NeuralNetworkModel(double learningRate = 0.1);
    
    // Add a layer to the network
    void addLayer(size_t inputSize, size_t outputSize, ActivationType activation = ActivationType::SIGMOID);
    
    // Forward pass through all layers
    Matrix predict(const Matrix& input);
    
    // Train the network with a single sample
    void train(const Matrix& input, const Matrix& target);
    
    // Train the network with multiple samples (batch training)
    void trainBatch(const std::vector<Matrix>& inputs, const std::vector<Matrix>& targets, size_t epochs);
    
    // Compute loss (Mean Squared Error)
    double computeMSE(const Matrix& prediction, const Matrix& target) const;
    
    // Compute cross-entropy loss
    double computeCrossEntropy(const Matrix& prediction, const Matrix& target) const;
    
    // Getters
    size_t getLayerCount() const { return layers.size(); }
    const Layer& getLayer(size_t index) const;
    double getLearningRate() const { return learningRate; }
    
    // Setters
    void setLearningRate(double lr) { learningRate = lr; }
};

} // namespace NeuralNetwork

#endif // NEURALNETWORK_NEURALNETWORK_H
