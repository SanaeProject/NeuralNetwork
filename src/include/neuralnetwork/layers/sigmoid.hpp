#ifndef SANAE_NEURALNETWORK_SIGMOID_HPP
#define SANAE_NEURALNETWORK_SIGMOID_HPP

#include <algorithm>
#include <execution>
#include <math.h>
#include <iostream>
#include "layerbase.hpp"
#include "../../matrix/matrix" // MatrixクラスとStdExecPolicyコンセプト

// シグモイドレイヤー
template<typename ty, typename ExecPolicy = std::execution::parallel_unsequenced_policy>
requires StdExecPolicy<ExecPolicy>
class Sigmoid : public LayerBase<ty> {
private:
    Matrix<ty> _out; // 出力の保存用

public:
    /**
     * 前向き伝播
     * @param in 入力
     * @return 出力
     * @note out = 1 / (1 + exp(-in))
     */
    Matrix<ty> forward(const Matrix<ty>& in) override{
        try{
            Matrix<ty> out = in;
            out.apply([](ty x) { return 1 / (1 + exp(-x)); }, ExecPolicy{});
            
            this->_out = out; // 出力を保存
            return out;
        }
        catch(const std::exception& e){
            std::cerr << "Error in Sigmoid forward: " << e.what() << std::endl;
            throw;
        }
    }
    /**
     * 逆伝播
     * @param dout 出力の勾配
     * @return 入力の勾配
     * @note dx = dout ⊙ (out ⊙ (1 - out))
     */
    Matrix<ty> backward(const Matrix<ty>& dout) override{
        try{
            // 保存しておいた出力 _out からシグモイドの導関数 out * (1 - out) を要素ごとに計算
            Matrix<ty> dx = this->_out;
            dx.apply([](ty y) { return y * (static_cast<ty>(1) - y); }, ExecPolicy{});
            // dx: out ⊙ (1 - out) に dout を要素ごとに掛ける
            dx = dx.hadamard_mul(dout, ExecPolicy{});
            return dx;
        }
        catch(const std::exception& e){
            std::cerr << "Error in Sigmoid backward: " << e.what() << std::endl;
            throw;
        }
    }
};

#endif //SANAE_NEURALNETWORK_SIGMOID_HPP