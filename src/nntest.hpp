#ifndef SANAE_NNTEST_HPP
#define SANAE_NNTEST_HPP

#include <cstddef>
#include <iostream>
#include <limits>
#include "./include/neuralnetwork/neuralnetwork.hpp"
#include "include/neuralnetwork/layers/affine.hpp"
#include "include/neuralnetwork/layers/relu.hpp"
#include "include/neuralnetwork/layers/batchnormalization.hpp"
#include "include/neuralnetwork/layers/softmaxwithloss.hpp"

void run_nntest() {
    using MyLayers = LayerPack<
        Affine<float>,
        BatchNormalization<float>,
        ReLU<float>,
        Affine<float>,
        SoftmaxWithLoss<float>
    >;
    size_t batch_size = 15;
    
    NeuralNetwork<float, MyLayers> a(2,4,2,0.1f);

    // XOR問題の学習 最大学習回数10000回
    for(size_t i = 0;i<10000;i++){
        Matrix<float> x(batch_size, 2, [&]() { return std::rand() % 2; }); // 0 or 1
        Matrix<float> t(batch_size, 2, [&]() { return 0.0f; });
        for (size_t j = 0; j < batch_size; j++) {
            bool d1 = x(j, 0);
            bool d2 = x(j, 1);
            bool label = d1 ^ d2; // XOR

            t(j, label ? 1 : 0) = 1.0f; // 正解ラベルをone-hotエンコード
        }
        try{
            if(i % 100 == 0){
                double loss = a.learn<true>(x, t);
                std::cout << "loss: " << loss << std::endl;

                if(loss < 1e-3){
                    std::cout << "Converged at iteration " << i << " with loss " << loss << std::endl;
                    break;
                }
            }
            else {
                a.learn<false>(x, t);
            }
        }catch(const std::exception& e){
            std::cout << e.what();
        }
    }

    while(true) {
        std::cout << "Enter two binary inputs (0 or 1) separated by space (or '-1 -1' to quit): ";
        int d1, d2;

        if (!(std::cin >> d1 >> d2)) {
            if (std::cin.eof()) {
                std::cout << "Exiting..." << std::endl;
                break;
            }

            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid input. Please enter two integers (0 or 1)." << std::endl;
            continue;
        }

        if (d1 == -1 && d2 == -1) {
            std::cout << "Exiting..." << std::endl;
            break;
        }

        if (d1 < 0 || d1 > 1 || d2 < 0 || d2 > 1) {
            std::cout << "Invalid input. Please enter 0 or 1." << std::endl;
            continue;
        }
        
        Matrix<float> x(1, 2);
        x(0, 0) = static_cast<float>(d1);
        x(0, 1) = static_cast<float>(d2);

        Matrix<float> output = a.predict(x);
        std::cout << "Predicted probabilities: " << d1 << " XOR " << d2 << " = " << (output.data()[0] < output.data()[1] ? "true" : "false") << std::endl;
    }
}

#endif // SANAE_NNTEST_HPP