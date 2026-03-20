#ifndef SANAE_NEURALNETWORK_AFFINE_HPP
#define SANAE_NEURALNETWORK_AFFINE_HPP

#include "layerbase.hpp"
#include "../../matrix/matrix"
#include "optimizer.hpp"
#include <execution>
#include <iostream>
#include <math.h>
#include <random>

template<typename ty, bool use_blas = true, typename OptimizerType = SGD<ty>, typename ExecType = std::execution::sequenced_policy>
requires DerivedOptimizer<OptimizerType, ty> && StdExecPolicy<ExecType>
class Affine : public LayerBase<ty> {
private:
    Matrix<ty> _in; // (batch, in_dim)
    Matrix<ty> _w;  // (in_dim, out_dim)
    Matrix<ty> _b;  // (1, out_dim)

    ty _learning_rate = 0.01f;

public:
    OptimizerType optimizer;

    Affine(size_t input_size, size_t output_size, ty lr = 0.01f, uint32_t seed = std::random_device{}())
        : _w(input_size, output_size),
          _b(1, output_size),
          optimizer(_w, _b, lr)
    {
        std::default_random_engine engine(seed);
        std::uniform_real_distribution<ty> dist(0, 1.0 / std::sqrt(input_size));

        _w = Matrix<ty>(input_size, output_size, [&](){ return dist(engine); });
        _b = Matrix<ty>(1, output_size, [&](){ return dist(engine); });
    }

    Matrix<ty> forward(const Matrix<ty>& in) override {
        _in = in; // (batch, in_dim)

        Matrix<ty> out = in;

        try{
            out.template matrix_mul<use_blas>(_w);
            out.apply_row(_b.data(), std::plus<ty>(), ExecType{});

            return out; // (batch, out_dim)
        } catch (const std::exception& e) {
            std::cerr << "Error in Affine::forward: " << e.what() << std::endl;
            throw;
        }
    }
    Matrix<ty> backward(const Matrix<ty>& dout) override {
        // dx = dout * W^T
        Matrix<ty> wt = _w.transpose_copy();
        Matrix<ty> dx = dout.template matrix_mul_copy<use_blas>(wt);

        // dW = X^T * dout
        Matrix<ty> in_t = _in.transpose_copy();
        Matrix<ty> dw = in_t.template matrix_mul_copy<use_blas>(dout);

        // db = sum(dout, axis=0)
        Matrix<ty> db = dout.sum_rows(); // (1, out_dim)

        optimizer.optimize(dw, db);
        return dx;
    }
};

#endif //SANAE_NEURALNETWORK_AFFINE_HPP