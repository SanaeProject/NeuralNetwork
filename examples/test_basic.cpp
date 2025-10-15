#include "NeuralNetwork.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

void testMatrixCreation() {
    std::cout << "Testing matrix creation..." << std::endl;
    
    NeuralNetwork::Matrix m1(2, 3);
    assert(m1.getRows() == 2);
    assert(m1.getCols() == 3);
    
    NeuralNetwork::Matrix m2(2, 2, 5.0);
    assert(m2(0, 0) == 5.0);
    assert(m2(1, 1) == 5.0);
    
    std::cout << "✓ Matrix creation tests passed" << std::endl;
}

void testMatrixOperations() {
    std::cout << "Testing matrix operations..." << std::endl;
    
    NeuralNetwork::Matrix A(2, 2, {1.0, 2.0, 3.0, 4.0});
    NeuralNetwork::Matrix B(2, 2, {5.0, 6.0, 7.0, 8.0});
    
    // Test addition
    auto C = A + B;
    assert(C(0, 0) == 6.0);
    assert(C(1, 1) == 12.0);
    
    // Test subtraction
    auto D = B - A;
    assert(D(0, 0) == 4.0);
    assert(D(1, 1) == 4.0);
    
    // Test scalar multiplication
    auto E = A * 2.0;
    assert(E(0, 0) == 2.0);
    assert(E(1, 1) == 8.0);
    
    // Test matrix multiplication
    auto F = A * B;
    assert(F(0, 0) == 19.0);  // 1*5 + 2*7
    assert(F(0, 1) == 22.0);  // 1*6 + 2*8
    assert(F(1, 0) == 43.0);  // 3*5 + 4*7
    assert(F(1, 1) == 50.0);  // 3*6 + 4*8
    
    // Test transpose
    auto G = A.transpose();
    assert(G.getRows() == 2);
    assert(G.getCols() == 2);
    assert(G(0, 0) == 1.0);
    assert(G(0, 1) == 3.0);
    assert(G(1, 0) == 2.0);
    assert(G(1, 1) == 4.0);
    
    std::cout << "✓ Matrix operation tests passed" << std::endl;
}

void testActivationFunctions() {
    std::cout << "Testing activation functions..." << std::endl;
    
    NeuralNetwork::Matrix input(2, 1, {0.0, 1.0});
    
    // Test sigmoid
    auto sigmoid_out = NeuralNetwork::Activation::sigmoid(input);
    assert(std::abs(sigmoid_out(0, 0) - 0.5) < 0.01);
    assert(std::abs(sigmoid_out(1, 0) - 0.7310) < 0.01);
    
    // Test ReLU
    NeuralNetwork::Matrix input2(2, 1, {-1.0, 1.0});
    auto relu_out = NeuralNetwork::Activation::relu(input2);
    assert(relu_out(0, 0) == 0.0);
    assert(relu_out(1, 0) == 1.0);
    
    // Test tanh
    auto tanh_out = NeuralNetwork::Activation::tanh(input);
    assert(std::abs(tanh_out(0, 0) - 0.0) < 0.01);
    assert(std::abs(tanh_out(1, 0) - 0.7615) < 0.01);
    
    std::cout << "✓ Activation function tests passed" << std::endl;
}

void testNeuralNetwork() {
    std::cout << "Testing neural network..." << std::endl;
    
    NeuralNetwork::NeuralNetworkModel nn(0.1);
    nn.addLayer(2, 3, NeuralNetwork::ActivationType::SIGMOID);
    nn.addLayer(3, 1, NeuralNetwork::ActivationType::SIGMOID);
    
    assert(nn.getLayerCount() == 2);
    
    NeuralNetwork::Matrix input(2, 1, {1.0, 0.0});
    NeuralNetwork::Matrix output = nn.predict(input);
    
    assert(output.getRows() == 1);
    assert(output.getCols() == 1);
    assert(output(0, 0) >= 0.0 && output(0, 0) <= 1.0);
    
    std::cout << "✓ Neural network tests passed" << std::endl;
}

int main() {
    std::cout << "=== Running NeuralNetwork Library Tests ===" << std::endl << std::endl;
    
    try {
        testMatrixCreation();
        testMatrixOperations();
        testActivationFunctions();
        testNeuralNetwork();
        
        std::cout << std::endl << "=== All tests passed successfully! ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
