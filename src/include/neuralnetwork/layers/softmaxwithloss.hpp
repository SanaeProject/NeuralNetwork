#ifndef SANAE_NEURALNETWORK_SOFTMAXWITHLOSS_HPP
#define SANAE_NEURALNETWORK_SOFTMAXWITHLOSS_HPP

#include <algorithm>
#include <execution>
#include <numeric>
#include <math.h>
#include <iostream>
#include "layerbase.hpp"
#include "../../matrix/matrix" // MatrixクラスとStdExecPolicyコンセプト

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
    Matrix<ty> forward(const Matrix<ty>& in) override {
        // in: (batch, classes)
        Matrix<ty> out = in;

        // 行ごとに softmax
        for (size_t i = 0; i < out.rows(); ++i) {
            ty* row = out.get_row_ptr(i);

            ty max_val = *std::max_element(row, row + out.cols());
            std::transform(row, row + out.cols(), row,
                           [max_val](ty x) { return std::exp(x - max_val); });

            ty sum = std::accumulate(row, row + out.cols(), static_cast<ty>(0));
            std::transform(row, row + out.cols(), row,
                           [sum](ty x) { return x / sum; });
        }

        _out = out;
        return out;
    }
    /**
     * 逆伝播
     * @param t 教師データ
     * @return 入力の勾配
     * @note dx = out - t
     */
    Matrix<ty> backward(const Matrix<ty>& t) override{
        try{
            // dx = (y - t) / batch_size
            Matrix<ty> dx = _out;
            dx.sub(t, ExecPolicy{});

            ty batch_size = static_cast<ty>(dx.rows());
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

        return static_cast<double>(total / _out.rows()); // バッチ平均
    }
};

#endif //SANAE_NEURALNETWORK_SOFTMAXWITHLOSS_HPP