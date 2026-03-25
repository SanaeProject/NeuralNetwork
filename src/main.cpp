#include <cstddef>
#include <execution>
#include <iostream>
#include "./nnlearncomplete.hpp"
#include "./include/neuralnetwork/neuralnetwork.hpp"
#include "include/neuralnetwork/layers/affine.hpp"
#include "include/neuralnetwork/layers/relu.hpp"
#include "include/neuralnetwork/layers/softmaxwithloss.hpp"

void run_matrix_tests();
void run_layertest();

int main() {
    using MyLayers = LayerPack<
        Affine<float>,
        ReLU<float>,
        Affine<float>,
        SoftmaxWithLoss<float>
    >;
    size_t batch_size = 1;
    
    NeuralNetwork<float, MyLayers> a(2,4,2,0.1f);

    for(size_t i = 0;i<1000;i++){
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
                std::cout << "loss: " << a.learn<true>(x, t) << std::endl;
            }
            else {
                a.learn<false>(x, t);
            }
        }catch(std::exception e){
            std::cout << e.what();
        }
    }

    return 0;
}