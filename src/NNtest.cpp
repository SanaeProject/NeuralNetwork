#include <iostream>
#include "include/matrix/matrix"

#include "include/layers/affine.hpp"
#include "include/layers/relu.hpp"
#include "include/layers/sigmoid.hpp"
#include "include/layers/softmaxwithloss.hpp"

void nntest() {
    Affine<float> affine1(2, 4);
    ReLU<float> relu1;

    Affine<float> affine2(4, 2);
    ReLU<float> relu2;

    SoftmaxWithLoss<float> softmaxwithloss;

    affine1.learning_rate = 0.3f;
    affine2.learning_rate = 0.3f;

    auto learn = [&](const Matrix<float>& x, const Matrix<float>& t) {

        auto out1 = affine1.forward(x);
        auto out2 = relu1.forward(out1);

        auto out3 = affine2.forward(out2);
        auto out4 = relu2.forward(out3);

        auto out5 = softmaxwithloss.forward(out4);

        auto dout5 = softmaxwithloss.backward(t);
        auto dout4 = relu2.backward(dout5);
        auto dout3 = affine2.backward(dout4);

        auto dout2 = relu1.backward(dout3);
        auto dout1 = affine1.backward(dout2);
    };

    auto predict = [&](const Matrix<float>& x) {

        auto out1 = affine1.forward(x);
        auto out2 = relu1.forward(out1);

        auto out3 = affine2.forward(out2);
        auto out4 = relu2.forward(out3);

        return out4;
    };

    for(int i = 0; i < 2000; i++) {

        bool d1 = rand() % 2;
        bool d2 = rand() % 2;
        bool label = d1 ^ d2;

        Matrix<float> x({{(float)d1, (float)d2}});
        Matrix<float> t({{(float)label, (float)!label}});

        learn(x, t);
    }

    for(int i = 0; i < 10; i++) {

        bool d1 = rand() % 2;
        bool d2 = rand() % 2;
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