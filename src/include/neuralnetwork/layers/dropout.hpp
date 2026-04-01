#ifndef SANAE_NEURALNETWORK_DROPOUT_HPP
#define SANAE_NEURALNETWORK_DROPOUT_HPP

#include <algorithm>
#include <cstdint>
#include <execution>
#include <math.h>
#include <iostream>
#include <random>
#include "layerbase.hpp"
#include "../../matrix/matrix" // MatrixクラスとStdExecPolicyコンセプト

// ドロップアウトレイヤー
template<typename ty, typename ExecPolicy = std::execution::sequenced_policy>
requires StdExecPolicy<ExecPolicy>
class Dropout : public LayerBase<ty> {
private:
    Matrix<ty> _mask; // ドロップアウトマスク
    uint32_t _seed; // 乱数シード
    ty _dropout_ratio;
public:
    static constexpr std::string_view name() { return "Dropout"; }
    
    Dropout(ty dropout_ratio = 0.5f, uint32_t seed = std::random_device{}()) {
        if (dropout_ratio < 0.0f || dropout_ratio >= 1.0f) {
            throw std::invalid_argument("Dropout ratio must be in the range [0.0, 1.0).");
        }

        this->_dropout_ratio = dropout_ratio;
    }
    
    /**
     * 前向き伝播
     * @param in 入力
     * @return 出力
     * @note out = in ⊙ mask (学習時), out = in * (1 - dropout_ratio) (推論時)  
     */
    Matrix<ty> forward(const Matrix<ty>& in) override{
        try{
            if(this->training){
                std::default_random_engine engine(this->_seed);
                std::bernoulli_distribution dist(1.0 - this->_dropout_ratio);

                _mask = Matrix<ty>(in.rows(), in.cols(), [&](){ return dist(engine) ? static_cast<ty>(1) : static_cast<ty>(0); });
                return in.hadamard_mul(_mask, ExecPolicy{});
            }else{
                return in.template scalar_mul_copy<true>(1.0f - this->_dropout_ratio, ExecPolicy{});
            }
        }
        catch(const std::exception& e){
            std::cerr << "Error in DROPOUT forward: " << e.what() << std::endl;
            throw;
        }
    }
    /**
     * 逆伝播
     * @param dout 出力の勾配
     * @return 入力の勾配
     * @note dx = dout ⊙ mask
     */
    Matrix<ty> backward(const Matrix<ty>& dout) override{
        try{
            if(!this->training)
                throw std::runtime_error("Error in DROPOUT backward: backward called during inference mode.");

            return dout.hadamard_mul(_mask, ExecPolicy{});
        }
        catch(const std::exception& e){
            std::cerr << "Error in DROPOUT backward: " << e.what() << std::endl;
            throw;
        }
    }
};

#endif //SANAE_NEURALNETWORK_DROPOUT_HPP