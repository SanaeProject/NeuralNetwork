#ifndef SANAE_NEURALNETWORK_AFFINE_HPP
#define SANAE_NEURALNETWORK_AFFINE_HPP

#include "layerbase.hpp"
#include <math.h>
#include <random>

// アフィンレイヤー
template<typename ty, bool use_blas = true>
class Affine : public LayerBase<ty, use_blas> {
private:
    Matrix<ty> _in; // 入力の保存用
    Matrix<ty> _w;
    Matrix<ty> _b;

public:
    double learning_rate = 0.01;

    /**
     * コンストラクタ
     * @param input_size 入力の次元数
     * @param output_size 出力の次元数
     * @param seed 乱数生成器のシード
     */
    Affine(size_t input_size, size_t output_size, uint32_t seed = std::random_device{}()) {
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
        this->_in = in;
        Matrix<ty> out = in;
        out.matrix_mul<use_blas>(_w).add(_b);
        return out;
    }

    /**
     * 逆伝播
     * @param dout 出力の勾配
     * @return 入力の勾配
     * @note dw = in^T * dout * η, db = dout * η
     */
    Matrix<ty> backward(const Matrix<ty>& dout) override{
        // 入力に対する勾配の計算: dx = dout * W^T
        Matrix<ty> dx = dout.matrix_mul<use_blas>(this->_w.transpose());

        // 勾配の計算（パラメータ更新用）
        Matrix<ty> dw = this->_in.transpose().matrix_mul<use_blas>(dout).scalar_mul<use_blas>(this->learning_rate); // in^T * dout * η
        Matrix<ty> db = dout; // dout のコピーを作成
        db.scalar_mul<use_blas>(this->learning_rate); // dout * η

        // パラメータの更新
        this->_w.sub<use_blas>(dw);
        this->_b.sub<use_blas>(db);

        return dx; // 入力に対する勾配を返す
    }
};

#endif //SANAE_NEURALNETWORK_AFFINE_HPP