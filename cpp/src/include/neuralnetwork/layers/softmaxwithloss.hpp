#ifndef SANAE_NEURALNETWORK_SOFTMAXWITHLOSS_HPP
#define SANAE_NEURALNETWORK_SOFTMAXWITHLOSS_HPP

#include <algorithm>
#include <execution>
#include <numeric>
#include <math.h>
#include <iostream>
#include "layerbase.hpp"
#include <stdexcept>
#include "../../matrix/matrix" // MatrixクラスとStdExecPolicyコンセプト

// ソフトマックス with ロスレイヤー
template<typename ty, typename ExecPolicy = std::execution::sequenced_policy>
requires StdExecPolicy<ExecPolicy>
class SoftmaxWithLoss : public LayerBase<ty> {
private:
    Matrix<ty> _out; // 出力の保存用

public:
    static constexpr std::string_view name() { return "SoftmaxWithLoss"; }
    static constexpr bool has_loss = true; // loss関数を所有

    /**
     * 前向き伝播
     * @param in 入力
     * @return 出力
     * @note out = softmax(in)
     */
    Matrix<ty> forward(const Matrix<ty>& in) override {
        // in: (batch, classes)
        Matrix<ty> out = in;
        if (out.rows() == 0 || out.cols() == 0) {
            throw std::runtime_error("Error in SoftmaxWithLoss forward: input matrix is empty.");
        }

        const ExecPolicy policy = ExecPolicy{};

        // 行ごとに softmax を適用
        for (size_t i = 0; i < out.rows(); ++i) {
            ty* row = out.get_row_ptr(i);

            ty max_val = *std::max_element(policy, row, row + out.cols());
            std::transform(policy, row, row + out.cols(), row,
                           [max_val](ty x) { return std::exp(x - max_val); });

            ty sum = std::reduce(policy, row, row + out.cols(), static_cast<ty>(0));
            if(sum <= 0){
                throw std::runtime_error("Error in SoftmaxWithLoss forward: sum of exponentials is non-positive.");
            }

            std::transform(policy, row, row + out.cols(), row,
                           [sum](ty x) { return x / sum; });
        }

        _out = out;
        return out;
    }
    /**
     * 逆伝播
     * @param t 教師データ
     * @return 入力の勾配
     * @note dx = (y - t) / batch_size
     */
    Matrix<ty> backward(const Matrix<ty>& t) override{
        try{
            // dx = (y - t) / batch_size
            Matrix<ty> dx = _out;
            dx.sub(t, ExecPolicy{});

            ty batch_size = static_cast<ty>(dx.rows());
            if(batch_size == 0){
                throw std::runtime_error("Error in SoftmaxWithLoss backward: batch size is zero.");
            }

            dx.scalar_div(batch_size, ExecPolicy{});

            return dx;
        }
        catch(const std::exception& e){
            std::cerr << "Error in SoftmaxWithLoss backward: " << e.what() << std::endl;
            throw;
        }
    }

    /**
    * @param t 教師データ
    * @return ロス値
    * @note loss = -Σ(t_i * log(out_i + ε))
    */
    double loss(const Matrix<ty>& t) {
        ty epsilon = static_cast<ty>(1e-7);
        ty total = 0;

        for (size_t i = 0; i < _out.rows(); ++i) {
            for (size_t j = 0; j < _out.cols(); ++j) {
                ty y = std::max(_out(i, j), epsilon);
                total -= t(i, j) * std::log(y);
            }
        }

        if(_out.rows() == 0){
            throw std::runtime_error("Error in SoftmaxWithLoss loss calculation: batch size is zero.");
        }

        return static_cast<double>(total / _out.rows()); // バッチ平均
    }
};

#endif //SANAE_NEURALNETWORK_SOFTMAXWITHLOSS_HPP