#ifndef SANAE_NEURALNETWORK_BATCHNORMALIZATION_HPP
#define SANAE_NEURALNETWORK_BATCHNORMALIZATION_HPP

#include <algorithm>
#include <execution>
#include <cmath>
#include <stdexcept>
#include <vector>
#include "layerbase.hpp"
#include "../../matrix/matrix" // MatrixクラスとStdExecPolicyコンセプト

template<typename ty, typename ExecPolicy = std::execution::sequenced_policy>
requires StdExecPolicy<ExecPolicy>
class BatchNormalization : public LayerBase<ty> {
private:
    std::vector<ty> gamma;   // γ[j]
    std::vector<ty> beta;    // β[j]

    std::vector<ty> running_mean; // 推論用
    std::vector<ty> running_var;  // 推論用

    std::vector<ty> muB;      // 学習時のバッチ平均
    std::vector<ty> sigma2B;  // 学習時のバッチ分散
    std::vector<ty> inv;      // 1/sqrt(var+eps)

    Matrix<ty> xhat;          // 学習時のみ保持
    ty eps = static_cast<ty>(1e-7);
    ty momentum = static_cast<ty>(0.9);

public:
    ty lr = static_cast<ty>(0.01);

    /**
     * @brief 列数を設定する
     * @param cols 列数
     */
    void set_cols(size_t cols) {
        gamma.resize(cols, 1);
        beta.resize(cols, 0);

        running_mean.resize(cols, 0);
        running_var.resize(cols, 1);
    }

    /**
     * @brief 順伝播
     * @param in 入力データ
     * @return 出力データ
     * @note 学習時: y = γ[j] * (x - muB) / sqrt(sigma2B + eps) + β[j]
     * @note 推論時: y = γ[j] * (x - running_mean) / sqrt(running_var + eps) + β[j]
     */
    Matrix<ty> forward(const Matrix<ty>& in) override {
        const size_t rows = in.rows();
        const size_t cols = in.cols();

        Matrix<ty> out(rows, cols);

        // 学習
        if (this->training) {
            muB = in.sum_rows().data();
            for (auto& v : muB) v /= rows;

            sigma2B = in.apply_row_copy(muB, [](ty a, ty b){ return (a - b)*(a - b); }).sum_rows().data();
            for (auto& v : sigma2B) v /= rows;

            // running update
            for (size_t j = 0; j < cols; j++) {
                running_mean[j] = momentum * running_mean[j] + (1 - momentum) * muB[j];
                running_var[j]  = momentum * running_var[j]  + (1 - momentum) * sigma2B[j];
            }

            // xhat = (x - muB) / sqrt(sigma2B + eps)
            inv.resize(cols);
            for (size_t j = 0; j < cols; j++)
                inv[j] = static_cast<ty>(1) / std::sqrt(sigma2B[j] + eps);

            xhat = in.apply_row_copy(muB, [](ty a, ty b){ return a - b; });
            xhat.apply_row(inv, [](ty a, ty b){ return a * b; });

            // y = γ[j] * xhat + β[j]
            for (size_t i = 0; i < rows; i++)
                for (size_t j = 0; j < cols; j++)
                    out(i,j) = gamma[j] * xhat(i,j) + beta[j];

        } else {
            // 推論
            for (size_t j = 0; j < cols; j++)
                inv[j] = static_cast<ty>(1) / std::sqrt(running_var[j] + eps);

            for (size_t i = 0; i < rows; i++)
                for (size_t j = 0; j < cols; j++) {
                    ty xhat_ij = (in(i,j) - running_mean[j]) * inv[j];
                    out(i,j) = gamma[j] * xhat_ij + beta[j];
                }
        }

        return out;
    }

    /**
     * @brief 逆伝播
     * @param dout 出力の勾配
     * @return 入力の勾配
     * @note dx = γ[j] * (1.0 / rows) * inv[j] * (rows * dout(i,j) - sum_dout - xhat(i,j) * sum_dout_xhat)
     * @note dγ[j] = Σ(dout(i,j) * xhat(i,j))
     * @note dβ[j] = Σ(dout(i,j))
     */
    Matrix<ty> backward(const Matrix<ty>& dout) override {
        if (!this->training) {
            throw std::runtime_error("Error in BatchNormalization backward: backward pass is not defined during inference.");
        }

        const size_t rows = dout.rows();
        const size_t cols = dout.cols();
        Matrix<ty> dx(rows, cols);

        // dγ[j], dβ[j]
        std::vector<ty> dgamma(cols, 0), dbeta(cols, 0);

        for (size_t j = 0; j < cols; j++) {
            for (size_t i = 0; i < rows; i++) {
                dgamma[j] += dout(i,j) * xhat(i,j);
                dbeta[j]  += dout(i,j);
            }
        }

        // update γ, β
        for (size_t j = 0; j < cols; j++) {
            gamma[j] -= lr * dgamma[j];
            beta[j]  -= lr * dbeta[j];
        }

        // dx
        for (size_t j = 0; j < cols; j++) {
            ty inv_std = inv[j];

            ty sum_dout = 0;
            ty sum_dout_xhat = 0;

            for (size_t i = 0; i < rows; i++) {
                sum_dout += dout(i,j);
                sum_dout_xhat += dout(i,j) * xhat(i,j);
            }

            for (size_t i = 0; i < rows; i++) {
                dx(i,j) =
                    gamma[j] * (1.0 / rows) * inv_std *
                    (rows * dout(i,j) - sum_dout - xhat(i,j) * sum_dout_xhat);
            }
        }

        return dx;
    }
};

#endif //SANAE_NEURALNETWORK_BATCHNORMALIZATION_HPP