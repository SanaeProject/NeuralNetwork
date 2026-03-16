#ifndef SANAE_NEURALNETWORK_RELU_HPP
#define SANAE_NEURALNETWORK_RELU_HPP

#include <algorithm>
#include <execution>
#include <math.h>
#include "layerbase.hpp"
#include "../matrix/matrix" // MatrixクラスとStdExecPolicyコンセプト

// ReLUレイヤー
template<typename ty, typename ExecPolicy = std::execution::parallel_unsequenced_policy>
requires StdExecPolicy<ExecPolicy>
class ReLU : public LayerBase<ty> {
private:
    Matrix<ty> _in; // 入力の保存用

public:
    double learning_rate = 0.01;

    /**
     * 前向き伝播
     * @param in 入力
     * @return 出力
     * @note out = max(0, in)
     */
    Matrix<ty> forward(const Matrix<ty>& in) override{
        this->_in = in; // 入力を保存
        Matrix<ty> out = in;
        out.apply([](ty x) { return std::max(static_cast<ty>(0), x); }, ExecPolicy{});
        
        return out;
    }
    /**
     * 逆伝播
     * @param dout 出力の勾配
     * @return 入力の勾配
     * @note dx = dout ⊙ (in > 0 ? 1 : 0)
     */
    Matrix<ty> backward(const Matrix<ty>& dout) override{
        // 保存しておいた入力 _in からReLUの導関数 (in > 0 ? 1 : 0) を要素ごとに計算
        Matrix<ty> dx = this->_in;
        dx.apply([](ty x) { return x > static_cast<ty>(0) ? static_cast<ty>(1) : static_cast<ty>(0); }, ExecPolicy{});
        // dx: (in > 0 ? 1 : 0) に dout を要素ごとに掛ける
        dx = dx.hadamard_mul(dout);
        return dx;
    }
};

#endif //SANAE_NEURALNETWORK_RELU_HPP