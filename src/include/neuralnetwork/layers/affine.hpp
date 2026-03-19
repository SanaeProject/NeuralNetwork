#ifndef SANAE_NEURALNETWORK_AFFINE_HPP
#define SANAE_NEURALNETWORK_AFFINE_HPP

#include "layerbase.hpp"
#include "../../matrix/matrix"
#include "optimizer.hpp"
#include <execution>
#include <iostream>
#include <math.h>
#include <random>

// affine layer wx+in
template<typename ty, bool use_blas = true, typename OptimizerType = SGD<ty>>
requires DerivedOptimizer<OptimizerType, ty>
class Affine : public LayerBase<ty> {
private:
    Matrix<ty> _in; // 入力の保存用
    Matrix<ty> _w;
    Matrix<ty> _b;
    ty _learning_rate = 0.01f;

public:
    OptimizerType optimizer;

    /**
     * 学習率を設定する
     * @param lr 学習率
     */
    void set_learning_rate(ty lr) {
        this->_learning_rate = lr;
        optimizer.set_learning_rate(lr);
    }

    /**
     * コンストラクタ
     * @param input_size 入力の次元数
     * @param output_size 出力の次元数
     * @param lr 学習率
     * @param seed 乱数生成器のシード
     */
    Affine(size_t input_size, size_t output_size, ty lr = 0.01f, uint32_t seed = std::random_device{}())
        : _w(input_size, output_size), _b(1, output_size),
          optimizer(_w, _b, lr)
    {
        std::default_random_engine engine(seed);
        std::uniform_real_distribution<ty> dist(0, (1.0 / std::sqrt(input_size))); // Xavier初期化の範囲

        // 重みとバイアスの初期化
        _w = Matrix<ty>(input_size, output_size, [&dist, &engine]() { return dist(engine); });
        _b = Matrix<ty>(1, output_size, [&dist, &engine]() { return dist(engine); });
    }

    /**
     * 前向き伝播
     * @param in 入力
     * @return 出力
     * @note out = in * w + b
     */
    Matrix<ty> forward(const Matrix<ty>& in) override{
        try{
            this->_in = in;
            Matrix<ty> out = in;
            out.template matrix_mul<use_blas>(_w).add(_b);
            return out;
        }
        catch(const std::exception& e){
            std::cerr << "Error in Affine forward: " << e.what() << std::endl;
            throw;
        }
    }

    /**
     * 逆伝播
     * @param dout 出力の勾配
     * @return 入力の勾配
     * @note dw = in^T * dout * η, db = dout * η
     */
    Matrix<ty> backward(const Matrix<ty>& dout) override{
        try{
            Matrix<ty> dout_copy = dout;

            // 入力に対する勾配の計算: dx = dout * W^T
            Matrix<ty> wt = this->_w;
            wt.transpose();
            Matrix<ty> dx = dout_copy.template matrix_mul<use_blas>(wt);

            // 勾配の計算（パラメータ更新用）
            Matrix<ty> in_t = this->_in;
            in_t.transpose();
            Matrix<ty> dw = in_t.template matrix_mul<use_blas>(dout); // in^T * dout
            Matrix<ty> db = dout; // dout のコピーを作成

            // パラメータの更新
            optimizer.optimize(dw, db);

            return dx; // 入力に対する勾配を返す
        }
        catch(const std::exception& e){
            std::cerr << "Error in Affine backward: " << e.what() << std::endl;
            throw;
        }
    }
};

#endif //SANAE_NEURALNETWORK_AFFINE_HPP