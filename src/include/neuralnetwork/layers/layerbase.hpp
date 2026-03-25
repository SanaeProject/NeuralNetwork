#ifndef NEURALNETWORK_LAYERBASE_HPP
#define NEURALNETWORK_LAYERBASE_HPP

#include "../../matrix/matrix"

// ベースレイヤー
template<typename ty>
class LayerBase {
public:
    static constexpr bool is_affine = false;
    static constexpr bool has_loss = false;

    virtual ~LayerBase() = default;
    virtual Matrix<ty> forward(const Matrix<ty>&) = 0;
    virtual Matrix<ty> backward(const Matrix<ty>&) = 0;
};

#endif //NEURALNETWORK_LAYERBASE_HPP