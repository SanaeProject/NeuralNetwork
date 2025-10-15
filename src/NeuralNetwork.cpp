#include "NeuralNetwork/NeuralNetwork.h"
#include <iostream>
#include <cmath>

namespace NeuralNetwork {

NeuralNetworkModel::NeuralNetworkModel(double learningRate)
    : learningRate(learningRate) {}

void NeuralNetworkModel::addLayer(size_t inputSize, size_t outputSize, ActivationType activation) {
    layers.push_back(std::make_unique<Layer>(inputSize, outputSize, activation));
}

Matrix NeuralNetworkModel::predict(const Matrix& input) {
    Matrix output = input;
    for (auto& layer : layers) {
        output = layer->forward(output);
    }
    return output;
}

void NeuralNetworkModel::train(const Matrix& input, const Matrix& target) {
    // Forward pass
    Matrix output = predict(input);
    
    // Compute initial gradient (derivative of loss with respect to output)
    // For MSE: gradient = 2 * (output - target)
    Matrix gradient = output - target;
    
    // Backward pass through all layers
    for (int i = layers.size() - 1; i >= 0; --i) {
        gradient = layers[i]->backward(gradient, learningRate);
    }
}

void NeuralNetworkModel::trainBatch(const std::vector<Matrix>& inputs, 
                                    const std::vector<Matrix>& targets, 
                                    size_t epochs) {
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("Number of inputs must match number of targets");
    }
    
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        double totalLoss = 0.0;
        
        for (size_t i = 0; i < inputs.size(); ++i) {
            // Train on single sample
            train(inputs[i], targets[i]);
            
            // Compute loss for monitoring
            Matrix prediction = predict(inputs[i]);
            totalLoss += computeMSE(prediction, targets[i]);
        }
        
        // Print progress every 100 epochs
        if ((epoch + 1) % 100 == 0) {
            double avgLoss = totalLoss / inputs.size();
            std::cout << "Epoch " << (epoch + 1) << "/" << epochs 
                     << " - Loss: " << avgLoss << std::endl;
        }
    }
}

double NeuralNetworkModel::computeMSE(const Matrix& prediction, const Matrix& target) const {
    if (prediction.getRows() != target.getRows() || 
        prediction.getCols() != target.getCols()) {
        throw std::invalid_argument("Prediction and target dimensions must match");
    }
    
    double mse = 0.0;
    for (size_t i = 0; i < prediction.getRows(); ++i) {
        for (size_t j = 0; j < prediction.getCols(); ++j) {
            double diff = prediction(i, j) - target(i, j);
            mse += diff * diff;
        }
    }
    
    return mse / (prediction.getRows() * prediction.getCols());
}

double NeuralNetworkModel::computeCrossEntropy(const Matrix& prediction, const Matrix& target) const {
    if (prediction.getRows() != target.getRows() || 
        prediction.getCols() != target.getCols()) {
        throw std::invalid_argument("Prediction and target dimensions must match");
    }
    
    double crossEntropy = 0.0;
    const double epsilon = 1e-10; // To avoid log(0)
    
    for (size_t i = 0; i < prediction.getRows(); ++i) {
        for (size_t j = 0; j < prediction.getCols(); ++j) {
            double p = std::max(epsilon, std::min(1.0 - epsilon, prediction(i, j)));
            crossEntropy -= target(i, j) * std::log(p);
        }
    }
    
    return crossEntropy / (prediction.getRows() * prediction.getCols());
}

const Layer& NeuralNetworkModel::getLayer(size_t index) const {
    if (index >= layers.size()) {
        throw std::out_of_range("Layer index out of range");
    }
    return *layers[index];
}

} // namespace NeuralNetwork
