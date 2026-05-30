#ifndef SANAE_NEURALNETWORK_TANH_HPP
#define SANAE_NEURALNETWORK_TANH_HPP

#include <algorithm>
#include <execution>
#include <cmath>
#include <iostream>
#include "layerbase.hpp"
#include "../../matrix/matrix" // MatrixクラスとStdExecPolicyコンセプト

// Tanhレイヤー
template<typename ty, typename ExecPolicy = std::execution::sequenced_policy>
requires StdExecPolicy<ExecPolicy>
class Tanh : public LayerBase<ty> {
private:
    Matrix<ty> _out; // 出力の保存用

public:
    static constexpr std::string_view name() { return "Tanh"; }
    
    /**
     * 前向き伝播
     * @param in 入力
     * @return 出力
     * @note out = tanh(x)
     */
    Matrix<ty> forward(const Matrix<ty>& in) override{
        try{
            this->_out = in;
            this->_out.apply([](ty x) { return std::tanh(x); }, ExecPolicy{});

            return this->_out;
        }
        catch(const std::exception& e){
            std::cerr << "Error in Tanh forward: " << e.what() << std::endl;
            throw;
        }
    }
    /**
     * 逆伝播
     * @param dout 出力の勾配
     * @return 入力の勾配
     * @note dx = (1 - tanh^2(x)) ⊙ dout
     */
    Matrix<ty> backward(const Matrix<ty>& dout) override{
        try{
            Matrix<ty> dx = this->_out;
            dx.apply([](ty y) { return static_cast<ty>(1) - (y * y); }, ExecPolicy{});
            dx.hadamard_mul(dout, ExecPolicy{});

            return dx;
        }
        catch(const std::exception& e){
            std::cerr << "Error in Tanh backward: " << e.what() << std::endl;
            throw;
        }
    }
};

#endif //SANAE_NEURALNETWORK_TANH_HPP