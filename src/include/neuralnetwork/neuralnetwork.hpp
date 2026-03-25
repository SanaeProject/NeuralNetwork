#ifndef SANAE_NEURALNETWORK_HPP
#define SANAE_NEURALNETWORK_HPP

#include "../matrix/matrix"
#include "./layers/layerbase.hpp"
#include "./layers/affine.hpp"

#include <memory>
#include <vector>
#include <concepts>

/**
 * LastType: 可変長テンプレートパラメータの最後の型を取得するための構造体
 * 最後のレイヤはloss関数を持つ必要があるため、最後の型はhas_loss == trueでなければならない
*/
template<class... L>
struct LastType;
template<class Head, class... Tail>
struct LastType<Head, Tail...> {
    using type = typename LastType<Tail...>::type;
};
template<class Last> requires (Last::has_loss == true) // 最後のレイヤはloss関数が必須
struct LastType<Last> {
    using type = Last;
};

/**
 * LayerPack: レイヤのパック。偶数層かつ4層以上必須
*/
template<class... Layers>
requires (sizeof...(Layers) % 2 == 0 && 4 <= sizeof...(Layers)) // 偶数層かつ4層以上必須
class LayerPack{};

/**
 * @tparam ty データ型
 * @tparam LayerPackT レイヤのパック
 * @tparam use_blas BLASを使用するかどうか
 * @tparam ExecType 実行ポリシーの型
 * @tparam DeviationType 標準偏差の計算方法の型
 * @tparam OptimizerType オプティマイザの型
 */
template<
    typename ty, 
    class LayerPackT
>
class NeuralNetwork {};

template<
    typename ty,
    class... Layers
>
class NeuralNetwork<ty, LayerPack<Layers...>>
{
    protected:
    std::vector<std::unique_ptr<LayerBase<ty>>> _layers;

    template<size_t size, size_t count>
    void _add_layer(size_t in_size, size_t hidden_size, size_t out_size, ty learning_rate, uint32_t seed){}

    template<size_t size, size_t count, class LayerHead, class... LayerTail> requires std::derived_from<LayerHead, LayerBase<ty>>
    void _add_layer(size_t in_size, size_t hidden_size, size_t out_size, ty learning_rate, uint32_t seed){
        // 最初のaffineレイヤ
        if constexpr (count == 0){
            _layers.emplace_back(std::make_unique<LayerHead>(in_size, hidden_size, learning_rate, seed));
        }
        else
        // 最後のaffineレイヤ
        if constexpr (sizeof...(LayerTail) == 1){
            _layers.emplace_back(std::make_unique<LayerHead>(hidden_size, out_size, learning_rate, seed));
        }
        else
        // 中間のaffineレイヤ
        if constexpr (LayerHead::is_affine){
            _layers.emplace_back(std::make_unique<LayerHead>(hidden_size, hidden_size, learning_rate, seed));
        }else{
            _layers.emplace_back(std::make_unique<LayerHead>());
        }

        this->_add_layer<size, count+1, LayerTail...>(in_size, hidden_size, out_size, learning_rate, seed);
    }

public:
    NeuralNetwork() = delete;
    NeuralNetwork(size_t in_size, size_t hidden_size, size_t out_size, ty learning_rate = 0.01f, uint32_t seed = std::random_device{}())
    {
        this->_add_layer<sizeof...(Layers), 0, Layers...>(in_size, hidden_size, out_size, learning_rate, seed);
    }

    template<bool use_loss = true>
    double learn(const Matrix<ty>& in, const Matrix<ty>& t){
        Matrix<ty> out = in;
        for(size_t i = 0; i < this->_layers.size(); i++){
            out = _layers.at(i)->forward(out);
        }

        out = t;
        for(size_t i = this->_layers.size(); i-- > 0; ){
            out = _layers.at(i)->backward(out);
        }

        if constexpr(use_loss){
            using Last = typename LastType<Layers...>::type;

            Last* last = dynamic_cast<Last*>(_layers.back().get());
            if (!last) {
                throw std::runtime_error("Last layer type mismatch");
            }

            return last->loss(t);
        }else{
            return 0;
        }
    }
};


#endif // SANAE_NEURALNETWORK_HPP