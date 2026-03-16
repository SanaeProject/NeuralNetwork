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
    Matrix<ty> _out; // 出力の保存用（ReLU適用後）

public:
    double learning_rate = 0.01;

    /**
     * 前向き伝播
     * @param in 入力
     * @return 出力
     * @note out = max(0, in)
     */
    Matrix<ty> forward(const Matrix<ty>& in) override{
        Matrix<ty> out = in;
        out.apply([](ty x) { return std::max(static_cast<ty>(0), x); }, ExecPolicy{});
        // ReLU 適用後の出力を保存（backward でマスク計算に使用）
        this->_out = out;
        
        return out;
    }
    /**
     * 逆伝播
     * @param dout 出力の勾配
     * @return 入力の勾配
     * @note dx = dout ⊙ (in > 0 ? 1 : 0)
     */
    Matrix<ty> backward(const Matrix<ty>& dout) override{
        // 保存しておいた出力 _out から ReLU の導関数 (in > 0 ? 1 : 0) を要素ごとに計算
        // ReLU では out = max(0, in) なので (in > 0) と (out > 0) は同値
        Matrix<ty> dx = this->_out;
        dx.apply([](ty x) { return x > static_cast<ty>(0) ? static_cast<ty>(1) : static_cast<ty>(0); }, ExecPolicy{});
        // dx: (in > 0 ? 1 : 0) に dout を要素ごとに掛ける
        dx.hadamard_mul(dout, ExecPolicy{});
        return dx;
    }
};

#endif //SANAE_NEURALNETWORK_RELU_HPP