#ifndef SANAE_NEURALNETWORK_SIGMOID_HPP
#define SANAE_NEURALNETWORK_SIGMOID_HPP

#include <algorithm>
#include <execution>
#include <math.h>
#include "layerbase.hpp"
#include "../matrix/matrix" // MatrixクラスとStdExecPolicyコンセプト

// シグモイドレイヤー
template<typename ty, typename ExecPolicy = std::execution::parallel_unsequenced_policy>
requires StdExecPolicy<ExecPolicy>
class Sigmoid : public LayerBase<ty> {
private:
    Matrix<ty> _out; // 出力の保存用

public:
    double learning_rate = 0.01;

    /**
     * 前向き伝播
     * @param in 入力
     * @return 出力
     * @note out = 1 / (1 + exp(-in))
     */
    Matrix<ty> forward(const Matrix<ty>& in) override{
        Matrix<ty> out = in;
        out.apply([](ty x) { return 1 / (1 + exp(-x)); }, ExecPolicy{});
        
        this->_out = out; // 出力を保存
        return out;
    }
    /**
     * 逆伝播
     * @param dout 出力の勾配
     * @return 入力の勾配
     * @note dx = dout * (out * (1 - out))
     */
    Matrix<ty> backward(const Matrix<ty>& dout) override{
        Matrix<ty> dx = dout;
        dx.apply([out = this->_out](ty x) { return x * out * (1 - out); }, ExecPolicy{});
        return dx;
    }
};

#endif //SANAE_NEURALNETWORK_SIGMOID_HPP