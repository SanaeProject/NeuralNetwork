#ifndef SANAE_NNLEARNCOMPLETE_HPP
#define SANAE_NNLEARNCOMPLETE_HPP

#include <execution>
#include <iostream>
#include <cstdlib>

#include "./include/neuralnetwork/layers/affine.hpp"
#include "./include/neuralnetwork/layers/relu.hpp"
#include "./include/neuralnetwork/layers/softmaxwithloss.hpp"
#include "./include/neuralnetwork/layers/optimizer.hpp"

template<typename execType = std::execution::parallel_unsequenced_policy, bool use_blas = true>
size_t run_nnlearn(float lr, uint32_t batch_size) {
    Affine<float, use_blas, execType, He> affine1(2, 4, lr);
    ReLU<float, execType> relu1;

    Affine<float, use_blas, execType, He> affine2(4, 2, lr);
    ReLU<float, execType> relu2;

    SoftmaxWithLoss<float, execType> softmaxwithloss;

    // 学習ループ
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

    // 予測関数
    auto predict = [&](const Matrix<float>& x) {
        auto out1 = affine1.forward(x);
        auto out2 = relu1.forward(out1);

        auto out3 = affine2.forward(out2);
        auto out5 = softmaxwithloss.forward(out3);

        return out5;
    };

    // 最大学習回数 10000回
    for(int i = 0; i < 10000; i++) {
        Matrix<float> x(batch_size, 2, [&]() { return std::rand() % 2; }); // 0 or 1
        Matrix<float> t(batch_size, 2, [&]() { return 0.0f; });
        for (size_t j = 0; j < batch_size; j++) {
            bool d1 = x(j, 0);
            bool d2 = x(j, 1);
            bool label = d1 ^ d2; // XOR

            t(j, label ? 1 : 0) = 1.0f; // 正解ラベルをone-hotエンコード
        }
        learn(x, t);

        if(i % 10 == 0) {
            double loss = softmaxwithloss.loss(t);
            // ロスが十分小さくなったら学習完了とみなす
            if(loss < 0.01) {
                return i; // 学習完了までのイテレーション数を返す
            }
        }
    }

    return 10000; // 最大イテレーション数を返す
}

#endif