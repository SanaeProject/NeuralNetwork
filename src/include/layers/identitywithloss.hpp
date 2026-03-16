#ifndef SANAE_NEURALNETWORK_IDENTITYWITHLOSS_HPP
#define SANAE_NEURALNETWORK_IDENTITYWITHLOSS_HPP

#include <algorithm>
#include <execution>
#include <math.h>
#include <iostream>
#include "layerbase.hpp"
#include "../matrix/matrix" // MatrixクラスとStdExecPolicyコンセプト

// 恒等函数 with ロスレイヤー
template<typename ty, typename ExecPolicy = std::execution::parallel_unsequenced_policy>
requires StdExecPolicy<ExecPolicy>
class IdentityWithLoss : public LayerBase<ty> {
private:
    Matrix<ty> _out; // 出力の保存用

public:
    /**
     * 前向き伝播
     * @param in 入力
     * @return 出力
     * @note out = in
     */
    Matrix<ty> forward(const Matrix<ty>& in) override{
        this->_out = in;
        return in;
    }

    /**
     * 
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
            std::cerr << "Error in IdentityWithLoss backward: " << e.what() << std::endl;
            throw;
        }
    }
};

#endif //SANAE_NEURALNETWORK_IDENTITYWITHLOSS_HPP