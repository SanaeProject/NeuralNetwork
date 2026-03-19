#ifndef SANAE_NEURALNETWORK_RELU_HPP
#define SANAE_NEURALNETWORK_RELU_HPP

#include <algorithm>
#include <execution>
#include <math.h>
#include <iostream>
#include "layerbase.hpp"
#include "../../matrix/matrix" // MatrixクラスとStdExecPolicyコンセプト

// ReLUレイヤー
template<typename ty, typename ExecPolicy = std::execution::parallel_unsequenced_policy>
requires StdExecPolicy<ExecPolicy>
class ReLU : public LayerBase<ty> {
private:
    Matrix<ty> _out; // 出力の保存用（ReLU適用後）

public:
    /**
     * 前向き伝播
     * @param in 入力
     * @return 出力
     * @note out = max(0, in)
     */
    Matrix<ty> forward(const Matrix<ty>& in) override{
        Matrix<ty> out = in;

        out.apply([](ty x) { 
            return x > 0 ? x : 0; 
        }, ExecPolicy{});
        _out = out; 

        return out;
    }
    /**
     * 逆伝播
     * @param dout 出力の勾配
     * @return 入力の勾配
     * @note dx = dout ⊙ (in > 0 ? 1 : 0)
     */
    Matrix<ty> backward(const Matrix<ty>& dout) override{
        Matrix<ty> dx = dout;
        Matrix<ty> out = this->_out; // ReLUの出力を保存しておいた_outから取得

        dx.hadamard_mul(
            out.apply(
                [](ty x){ return x > 0 ? 1 : 0; }, 
                ExecPolicy{}
            ),
            ExecPolicy{}
        );

        return dx;
    }
};

#endif //SANAE_NEURALNETWORK_RELU_HPP