#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

/**
 * @file NeuralNetwork.hpp
 * @brief Main header file for the NeuralNetwork library
 * 
 * This header includes all necessary components of the NeuralNetwork library.
 * Include this file to use the complete neural network functionality.
 * 
 * Example usage:
 * @code
 * #include "NeuralNetwork.hpp"
 * 
 * int main() {
 *     NeuralNetwork::NeuralNetworkModel nn(0.5);
 *     nn.addLayer(2, 4, NeuralNetwork::ActivationType::SIGMOID);
 *     nn.addLayer(4, 1, NeuralNetwork::ActivationType::SIGMOID);
 *     
 *     NeuralNetwork::Matrix input(2, 1, {0.0, 1.0});
 *     NeuralNetwork::Matrix target(1, 1, {1.0});
 *     
 *     nn.train(input, target);
 *     return 0;
 * }
 * @endcode
 */

#include "NeuralNetwork/Matrix.h"
#include "NeuralNetwork/ActivationFunctions.h"
#include "NeuralNetwork/Layer.h"
#include "NeuralNetwork/NeuralNetwork.h"

#endif // NEURALNETWORK_H
