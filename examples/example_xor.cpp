#include "NeuralNetwork.hpp"
#include <iostream>

int main() {
    std::cout << "=== Neural Network XOR Example ===" << std::endl;
    std::cout << "Training a neural network to learn XOR function" << std::endl << std::endl;
    
    // Create a neural network
    // Architecture: 2 inputs -> 4 hidden neurons -> 1 output
    NeuralNetwork::NeuralNetworkModel nn(0.5); // Learning rate = 0.5
    
    nn.addLayer(2, 4, NeuralNetwork::ActivationType::SIGMOID);  // Hidden layer
    nn.addLayer(4, 1, NeuralNetwork::ActivationType::SIGMOID);  // Output layer
    
    // Prepare XOR training data
    std::vector<NeuralNetwork::Matrix> inputs;
    std::vector<NeuralNetwork::Matrix> targets;
    
    // XOR truth table:
    // 0 XOR 0 = 0
    inputs.push_back(NeuralNetwork::Matrix(2, 1, {0.0, 0.0}));
    targets.push_back(NeuralNetwork::Matrix(1, 1, {0.0}));
    
    // 0 XOR 1 = 1
    inputs.push_back(NeuralNetwork::Matrix(2, 1, {0.0, 1.0}));
    targets.push_back(NeuralNetwork::Matrix(1, 1, {1.0}));
    
    // 1 XOR 0 = 1
    inputs.push_back(NeuralNetwork::Matrix(2, 1, {1.0, 0.0}));
    targets.push_back(NeuralNetwork::Matrix(1, 1, {1.0}));
    
    // 1 XOR 1 = 0
    inputs.push_back(NeuralNetwork::Matrix(2, 1, {1.0, 1.0}));
    targets.push_back(NeuralNetwork::Matrix(1, 1, {0.0}));
    
    // Train the network
    std::cout << "Training neural network..." << std::endl;
    nn.trainBatch(inputs, targets, 10000);
    
    // Test the network
    std::cout << std::endl << "=== Testing ===" << std::endl;
    for (size_t i = 0; i < inputs.size(); ++i) {
        NeuralNetwork::Matrix prediction = nn.predict(inputs[i]);
        
        std::cout << "Input: (" << inputs[i](0, 0) << ", " << inputs[i](1, 0) << ") -> ";
        std::cout << "Prediction: " << prediction(0, 0) << " (Target: " << targets[i](0, 0) << ")" << std::endl;
    }
    
    std::cout << std::endl << "=== Matrix Operations Example ===" << std::endl;
    
    // Demonstrate matrix operations
    NeuralNetwork::Matrix A(2, 2, {1.0, 2.0, 3.0, 4.0});
    NeuralNetwork::Matrix B(2, 2, {5.0, 6.0, 7.0, 8.0});
    
    std::cout << "Matrix A:" << std::endl;
    A.print();
    
    std::cout << std::endl << "Matrix B:" << std::endl;
    B.print();
    
    std::cout << std::endl << "A + B:" << std::endl;
    (A + B).print();
    
    std::cout << std::endl << "A * B (matrix multiplication):" << std::endl;
    (A * B).print();
    
    std::cout << std::endl << "A.transpose():" << std::endl;
    A.transpose().print();
    
    std::cout << std::endl << "A * 2.5 (scalar multiplication):" << std::endl;
    (A * 2.5).print();
    
    return 0;
}
