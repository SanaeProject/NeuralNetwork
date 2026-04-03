#ifndef SANAE_NEURALNETWORK_CONVOLUTION_HPP
#define SANAE_NEURALNETWORK_CONVOLUTION_HPP

#include <algorithm>
#include <execution>
#include <math.h>
#include <iostream>
#include "layerbase.hpp"
#include "../../matrix/matrix" // MatrixクラスとStdExecPolicyコンセプト

// 畳み込みレイヤー
template<typename ty, typename ExecPolicy = std::execution::sequenced_policy>
requires StdExecPolicy<ExecPolicy>
class Convolution : public LayerBase<ty> {
private:
    using Kernel = std::vector<Matrix<ty>>;

    const size_t _kernel_width;
    const size_t _kernel_height;

    const size_t _in_channel_size;
    const size_t _out_channel_size;

    Kernel kernel;

public:
    static constexpr std::string_view name() { return "Convolution"; }
    
    Convolution(size_t kernel_width, size_t kernel_height, size_t in_channel_size, size_t out_channel_size)
        : _kernel_width(kernel_width),
          _kernel_height(kernel_height),
          _in_channel_size(in_channel_size),
          _out_channel_size(out_channel_size)
    {
        for(size_t i = 0; i < ){

        }
    }

    /**
     * 前向き伝播
     * @param in 入力
     * @return 出力
     * @note out = 1 / (1 + exp(-in))
     */
    Matrix<ty> forward(const Matrix<ty>& in) override{
        try{


        }
        catch(const std::exception& e){
            std::cerr << "Error in Convolution forward: " << e.what() << std::endl;
            throw;
        }
    }
    /**
     * 逆伝播
     * @param dout 出力の勾配
     * @return 入力の勾配
     * @note dx = dout ⊙ (out ⊙ (1 - out))
     */
    Matrix<ty> backward(const Matrix<ty>& dout) override{
        try{


        }
        catch(const std::exception& e){
            std::cerr << "Error in Convolution backward: " << e.what() << std::endl;
            throw;
        }
    }
};

#endif //SANAE_NEURALNETWORK_CONVOLUTION_HPP