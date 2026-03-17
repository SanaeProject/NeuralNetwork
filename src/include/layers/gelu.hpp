#ifndef SANAE_NEURALNETWORK_GELU_HPP
#define SANAE_NEURALNETWORK_GELU_HPP

#include <algorithm>
#include <execution>
#include <math.h>
#include <iostream>
#include "layerbase.hpp"
#include "../matrix/matrix" // MatrixクラスとStdExecPolicyコンセプト

// GELUレイヤー
template<typename ty, typename ExecPolicy = std::execution::parallel_unsequenced_policy>
requires StdExecPolicy<ExecPolicy>
class GELU : public LayerBase<ty> {
private:
    Matrix<ty> _in; // 入力の保存用
    static constexpr ty c =  0.7978845608; // sqrt(2/π)

    /**
        y = 0.5x * (1+tanh(u)) = 0.5x+0.5x*tanh(u)
        u = sqrt(2/PI) * (x+0.044715*x^3)
        y = f(x)g(x)でdy/dx = f'(x)g(x)+f(x)g'(x)より
        dy/dx = {0.5(1+tanh(u))}+{0.5x*dtanh(u)/dx}
        dtanh(u)/du = 1 - tanh^2(u)
        du/dx = sqrt(2/PI) * (1+3*0.044715*x^2)
        dy/dx = {0.5(1+tanh(u))}+{0.5x*(1 - tanh^2(u))*sqrt(2/PI) * (1+3*0.044715*x^2)}
     */
    inline ty _gelu_grad(ty x) {
        const ty x2 = x * x;
        const ty u = c * (x + 0.044715 * x * x2);
        const ty t = std::tanh(u);

        return 0.5 * (1 + t)
            + 0.5 * x * (1 - t * t) * c * (1 + 0.134145 * x2);
    }
public:
    /**
     * 前向き伝播
     * @param in 入力
     * @return 出力
     * @note out = GELU(x) ≒ 0.5x {1+tanh(sqrt(2/π)*(x+0.044715*x^3))}
     */
    Matrix<ty> forward(const Matrix<ty>& in) override{
        try{
            Matrix<ty> out = in;
            this->_in = in; // 入力を保存

            out.apply([](ty x) {
                return 0.5 * x * (1 + std::tanh(c * (x + 0.044715 * x * x * x)));
             }, ExecPolicy{});

            return out;
        }
        catch(const std::exception& e){
            std::cerr << "Error in GELU forward: " << e.what() << std::endl;
            throw;
        }
    }
    /**
     * 逆伝播
     * @param dout 出力の勾配
     * @return 入力の勾配
     * @note dx = dout ⊙ (0.5 * (1 + tanh(u))+0.5 * x(1−tanh^2(u)) * c(1+0.134145 * x^2))
     * u = sqrt(2/PI) * (x+0.044715*x^3)
     */
    Matrix<ty> backward(const Matrix<ty>& dout) override{
        try{
            Matrix<ty> dx = this->_in;
            dx.apply([this](ty y) { return this->_gelu_grad(y); }, ExecPolicy{});
            dx = dx.hadamard_mul(dout, ExecPolicy{});
            return dx;
        }
        catch(const std::exception& e){
            std::cerr << "Error in GELU backward: " << e.what() << std::endl;
            throw;
        }
    }
};

#endif //SANAE_NEURALNETWORK_GELU_HPP