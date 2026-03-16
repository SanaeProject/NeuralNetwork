#ifndef NEURALNETWORK_LAYERBASE_HPP
#define NEURALNETWORK_LAYERBASE_HPP

#include "../matrix/matrix"

// ベースレイヤー
template<typename ty, bool use_blas = true>
class Layer_Base {
public:
    virtual Matrix<ty> forward(const Matrix<ty>&) = 0;
    virtual Matrix<ty> backward(const Matrix<ty>&) = 0;
};

#endif //NEURALNETWORK_LAYERBASE_HPP