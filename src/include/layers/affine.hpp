#ifndef SANAE_NEURALNETWORK_AFFINE_HPP
#define SANAE_NEURALNETWORK_AFFINE_HPP

#include "layerbase.hpp"

// アフィンレイヤー
template<typename ty, bool use_blas = true>
class Affine : public Layer_Base<ty, use_blas> {
private:
    Matrix<ty> _in; // 入力の保存用
    Matrix<ty> _w;
    Matrix<ty> _b;

public:
    double learning_rate = 0.01;

    Affine(size_t input_size, size_t output_size);

    /**
     * 前向き伝播
     * @param in 入力
     * @return 出力
     * @note out = in * w + b
     */
    Matrix<ty> forward(const Matrix<ty>& in) override{
        this->_in = in;
        return in.matrix_mul<use_blas>(_w).add(_b);
    }
    /**
     * 逆伝播
     * @param dout 出力の勾配
     * @return 入力の勾配
     * @note dw = in^T * dout * η, db = dout * η
     */
    Matrix<ty> backward(const Matrix<ty>& dout) override{
        // 勾配の計算
        Matrix<ty> dw = this->_in.transpose().matrix_mul<use_blas>(dout).scalar_mul<use_blas>(this->learning_rate); // in^T * dout * η
        Matrix<ty> db = dout.scalar_mul<use_blas>(this->learning_rate); // dout * η

        // パラメータの更新
        this->_w.sub<use_blas>(dw);
        this->_b.sub<use_blas>(db);

        return dw; // 入力に対する勾配を返す
    }
};

#endif //SANAE_NEURALNETWORK_AFFINE_HPP