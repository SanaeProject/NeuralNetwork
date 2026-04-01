#ifndef NEURALNETWORK_LAYERBASE_HPP
#define NEURALNETWORK_LAYERBASE_HPP

#include "../../matrix/matrix"
#include <string>

// ベースレイヤー
template<typename ty>
class LayerBase {
public:
    static constexpr bool is_affine = false;
    static constexpr bool has_loss = false;

    static constexpr std::string_view name() { return "LayerBase"; }

    bool training = true;
    virtual ~LayerBase() = default;
    virtual Matrix<ty> forward(const Matrix<ty>&) = 0;
    virtual Matrix<ty> backward(const Matrix<ty>&) = 0;
};

#endif //NEURALNETWORK_LAYERBASE_HPP