#include <iostream>
#include <cstdlib>

#include "./include/neuralnetwork/layers/affine.hpp"
#include "./include/neuralnetwork/layers/relu.hpp"
#include "./include/neuralnetwork/layers/softmaxwithloss.hpp"
#include "./include/neuralnetwork/layers/optimizer.hpp"

void run_layertest() {
    float learning_rate = 0.3f;
    Affine<float,true, SGD<float>> affine1(2, 4, learning_rate);
    ReLU<float> relu1;

    Affine<float,true, SGD<float>> affine2(4, 2, learning_rate);
    ReLU<float> relu2;

    SoftmaxWithLoss<float> softmaxwithloss;

    auto learn = [&](const Matrix<float>& x, const Matrix<float>& t) {
        auto out1 = affine1.forward(x);
        auto out2 = relu1.forward(out1);

        auto out3 = affine2.forward(out2);
        auto out5 = softmaxwithloss.forward(out3);
        auto dout5 = softmaxwithloss.backward(t);

        auto dout3 = affine2.backward(dout5);
        auto dout2 = relu1.backward(dout3);
        auto dout1 = affine1.backward(dout2);
    };

    auto predict = [&](const Matrix<float>& x) {
        auto out1 = affine1.forward(x);
        auto out2 = relu1.forward(out1);

        auto out3 = affine2.forward(out2);
        auto out5 = softmaxwithloss.forward(out3);

        return out5;
    };

    for(int i = 0; i < 2000; i++) {
        bool d1 = std::rand() % 2;
        bool d2 = std::rand() % 2;
        bool label = d1 ^ d2;

        Matrix<float> x({{(float)d1, (float)d2}});
        Matrix<float> t({{(float)label, (float)!label}});

        learn(x, t);

        if(i%200==0)
            std::cout << softmaxwithloss.loss(t) << std::endl;
    }

    for(int i = 0; i < 10; i++) {
        bool d1 = std::rand() % 2;
        bool d2 = std::rand() % 2;
        bool label = d1 ^ d2;

        Matrix<float> prediction = predict(Matrix<float>({{(float)d1, (float)d2}}));

        int result = prediction.data()[0] > prediction.data()[1] ? 1 : 0;

        std::cout
            << "Input: " << d1 << ", " << d2
            << " | Label: " << label
            << " | Prediction: " << result
            << std::endl;
    }
}