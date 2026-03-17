#ifndef SANAE_NEURALNETWORK_SOFTMAXWITHLOSS_HPP
#define SANAE_NEURALNETWORK_SOFTMAXWITHLOSS_HPP

#include <algorithm>
#include <execution>
#include <numeric>
#include <math.h>
#include <iostream>
#include "layerbase.hpp"
#include "../matrix/matrix" // MatrixクラスとStdExecPolicyコンセプト

// ソフトマックス with ロスレイヤー
template<typename ty, typename ExecPolicy = std::execution::parallel_unsequenced_policy>
requires StdExecPolicy<ExecPolicy>
class SoftmaxWithLoss : public LayerBase<ty> {
private:
    Matrix<ty> _out; // 出力の保存用

public:
    /**
     * 前向き伝播
     * @param in 入力
     * @return 出力
     * @note out = softmax(in)
     */
    Matrix<ty> forward(const Matrix<ty>& in) override{
        try{
        ty max_val = *std::max_element(in.data().begin(), in.data().end()); // 数値安定化のために最大値を引く
        Matrix<ty> out = in;
        out.apply([max_val](ty x) { return exp(x - max_val); }, ExecPolicy{});
        
        ty sum = std::accumulate(out.data().begin(), out.data().end(), static_cast<ty>(0));
        out.scalar_div(sum, ExecPolicy{});
        this->_out = out;
        return out;
        }
        catch(const std::exception& e){
            std::cerr << "Error in SoftmaxWithLoss forward: " << e.what() << std::endl;
            throw;
        }
    }
    /**
     * 逆伝播
     * @param t 教師データ
     * @return 入力の勾配
     * @note dx = out - t
     */
    Matrix<ty> backward(const Matrix<ty>& t) override{
        try{
            Matrix<ty> dx = this->_out;
            dx.sub(t, ExecPolicy{});
            return dx;
        }
        catch(const std::exception& e){
            std::cerr << "Error in SoftmaxWithLoss backward: " << e.what() << std::endl;
            throw;
        }
    }
};

#endif //SANAE_NEURALNETWORK_SOFTMAXWITHLOSS_HPP