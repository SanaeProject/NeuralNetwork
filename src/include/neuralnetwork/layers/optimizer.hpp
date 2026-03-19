#ifndef SANAE_NEURALNETWORK_OPTIMIZER_HPP
#define SANAE_NEURALNETWORK_OPTIMIZER_HPP

#include "../../matrix/matrix"
#include <execution>
#include <iostream>
#include <math.h>
#include <concepts>

template<typename ty>
class Optimizer {
protected:
    ty _learning_rate = 0.01f;
public:
    Optimizer(ty lr = 0.01f) {
        this->_learning_rate = lr;
    }
    void set_learning_rate(ty lr) {
        this->_learning_rate = lr;
    }

    virtual ~Optimizer() = default;
    virtual void optimize(Matrix<ty>&, Matrix<ty>&) = 0;
};

template<typename T, typename ty>
concept DerivedOptimizer = std::derived_from<T, Optimizer<ty>>;

template<typename ty, typename execPolicy = std::execution::sequenced_policy, bool use_blas = true>
requires StdExecPolicy<execPolicy>
class SGD : public Optimizer<ty>{
private:
    Matrix<ty>& _w;
    Matrix<ty>& _b;

public:
    SGD(Matrix<ty>& w, Matrix<ty>& b, ty learning_rate = 0.01f)
        : _w(w), _b(b), 
          Optimizer<ty>(learning_rate)
    {}

    inline void optimize(Matrix<ty>& dw, Matrix<ty>& db) override {
        try{
            // パラメータの更新
            this->_w = this->_w - dw.template scalar_mul_copy<use_blas>(this->_learning_rate, execPolicy{}); // w1 = w0 - dL/dw = w0 - in^T * dout * η
            this->_b = this->_b - db.template scalar_mul_copy<use_blas>(this->_learning_rate, execPolicy{}); // b1 = b0 - dL/db = b0 - dout * η
        }
        catch(const std::exception& e){
            std::cerr << "Error in SGD::optimize: " << e.what() << std::endl;
            throw;
        }
    } 
};
template<typename ty, typename execPolicy = std::execution::sequenced_policy, bool use_blas = true>
class Momentum: public Optimizer<ty> {
private:
    Matrix<ty>& _w;
    Matrix<ty>& _b;

    Matrix<ty> _vW;
    Matrix<ty> _vB;

    ty _momentum = 0.9f;

public:
    Momentum(Matrix<ty>& w, Matrix<ty>& b, ty learning_rate = 0.01f)
        : _w(w), _b(b),
          _vW(w.rows(), w.cols(), [](){return 0;}),
          _vB(b.rows(), b.cols(), [](){return 0;}),
          Optimizer<ty>(learning_rate)
    {}

    void set_momentum(ty momentum) {
        this->_momentum = momentum;
    }

    inline void optimize(Matrix<ty>& dw, Matrix<ty>& db) override {
        // vW = momentum * vW - lr * dW
        _vW = _vW.template scalar_mul_copy<use_blas>(this->_momentum, execPolicy{}) - dw.template scalar_mul_copy<use_blas>(this->_learning_rate, execPolicy{});
        _w  = _w + _vW;

        // vB = momentum * vB - lr * dB
        _vB = _vB.template scalar_mul_copy<use_blas>(this->_momentum, execPolicy{}) - db.template scalar_mul_copy<use_blas>(this->_learning_rate, execPolicy{});
        _b  = _b + _vB;
    }
};
template<typename ty, typename execPolicy = std::execution::sequenced_policy, bool use_blas = true>
class AdaGrad: public Optimizer<ty> {
private:
    Matrix<ty>& _w;
    Matrix<ty>& _b;
    Matrix<ty> _hw;
    Matrix<ty> _hb;

public:
    AdaGrad(Matrix<ty>& w, Matrix<ty>& b, ty learning_rate = 0.01f)
        : _w(w), _b(b), 
          _hw(w.rows(), w.cols(), [](){ return 0;}),
          _hb(b.rows(), b.cols(), [](){ return 0;}),
          Optimizer<ty>(learning_rate)
    {}

    inline void optimize(Matrix<ty>& dw, Matrix<ty>& db) override {
        try{
            // W更新
            // hw = hw + dw ⊙ dw
            this->_hw = this->_hw + dw.hadamard_mul_copy(dw, execPolicy{});

            // scale = 1 / sqrt(hw + ε)
            Matrix<ty> scale = this->_hw.apply_copy([](ty d){ return static_cast<ty>(1)/std::sqrt(d + 1e-8);}, execPolicy{});

            // update = scale ⊙ dw * η
            Matrix<ty> updateW = scale.hadamard_mul_copy(dw, execPolicy{}).template scalar_mul<use_blas>(this->_learning_rate, execPolicy{});

            // W更新
            // w = w - update
            this->_w = this->_w - updateW;

            // hb = hb + db ⊙ db
            this->_hb = this->_hb + db.hadamard_mul_copy(db, execPolicy{});

            // scale = 1 / sqrt(hb + ε)
            scale = this->_hb.apply_copy([](ty d){ return static_cast<ty>(1)/std::sqrt(d + 1e-8);}, execPolicy{});

            // update = scale ⊙ db * η
            Matrix<ty> updateB = scale.hadamard_mul_copy(db, execPolicy{}).template scalar_mul<use_blas>(this->_learning_rate, execPolicy{});

            this->_b = this->_b - updateB;
        }
        catch(const std::exception& e){
            std::cerr << "Error in AdaGrad optimize: " << e.what() << std::endl;
            throw;
        }
    }
};
template<typename ty, typename execPolicy = std::execution::sequenced_policy, bool use_blas = true>
requires StdExecPolicy<execPolicy>
class Adam : public Optimizer<ty>{
private:
    Matrix<ty>& _w;
    Matrix<ty> _wm, _wv;

    Matrix<ty>& _b;
    Matrix<ty> _bm, _bv;

    ty _momentum = 0.9f;
    ty _rms = 0.999f;
    size_t _time = 0;

public:
    Adam(Matrix<ty>& w, Matrix<ty>& b, ty learning_rate = 0.01f)
        : _w(w), _b(b), 

          _wm(w.rows(),w.cols(),[](){ return 0;}),
          _wv(w.rows(),w.cols(),[](){ return 0;}),

          _bm(b.rows(),b.cols(),[](){ return 0;}),
          _bv(b.rows(),b.cols(),[](){ return 0;}),
          Optimizer<ty>(learning_rate)
        {}

    void set_momentum(ty momentum) {
        this->_momentum = momentum;
    }
    void set_rms(ty rms) {
        this->_rms = rms;
    }

    inline void optimize(Matrix<ty>& dw, Matrix<ty>& db) override {
        try{
            this->_time += 1;

            // ---- W ----
            // m = β1*m + (1-β1)*dw
            {
                Matrix<ty> tmp_m = this->_wm;
                tmp_m.template scalar_mul<use_blas>(this->_momentum, execPolicy{}); // β1*m

                Matrix<ty> tmp_dw = dw;
                tmp_dw.template scalar_mul<use_blas>((1 - this->_momentum), execPolicy{}); // (1-β1)*dw

                this->_wm = tmp_m + tmp_dw;
            }

            // v = β2*v + (1-β2)*(dw ⊙ dw)
            {
                Matrix<ty> tmp_v = this->_wv;
                tmp_v.template scalar_mul<use_blas>(this->_rms, execPolicy{}); // β2*v

                Matrix<ty> dw_sq = dw;                          // copy
                dw_sq.hadamard_mul(dw, execPolicy{});           // dw ⊙ dw
                dw_sq.template scalar_mul<use_blas>((1 - this->_rms), execPolicy{}); // (1-β2)*(dw^2)

                this->_wv = tmp_v + dw_sq;
            }

            // m_hat = m / (1 - β1^t)
            Matrix<ty> m_hat = _wm;
            m_hat.template scalar_mul<use_blas>(1.0 / (1 - std::pow(this->_momentum, this->_time)), execPolicy{});

            // v_hat = v / (1 - β2^t)
            Matrix<ty> v_hat = _wv;
            v_hat.template scalar_mul<use_blas>(1.0 / (1 - std::pow(this->_rms, this->_time)), execPolicy{});

            // update = m_hat / (sqrt(v_hat) + ε)
            Matrix<ty> updateW = m_hat; // start from m_hat; will be scaled by 1 / (sqrt(v_hat) + ε)

            // divide by sqrt(v_hat) + ε
            {
                Matrix<ty> denom = v_hat;
                denom.apply([this](ty x){ return static_cast<ty>(1) / (std::sqrt(x) + 1e-8); }, execPolicy{});
                updateW.hadamard_mul(denom, execPolicy{});
            }

            updateW.template scalar_mul<use_blas>(this->_learning_rate, execPolicy{});
            _w.template sub<use_blas>(updateW, execPolicy{});

            // ---- B ----

            // m = β1*m + (1-β1)*db
            {
                Matrix<ty> tmp_m = this->_bm;
                tmp_m.template scalar_mul<use_blas>(this->_momentum, execPolicy{}); // β1*m

                Matrix<ty> tmp_db = db;
                tmp_db.template scalar_mul<use_blas>((1 - this->_momentum), execPolicy{}); // (1-β1)*db

                this->_bm = tmp_m + tmp_db;
            }

            // v = β2*v + (1-β2)*(db ⊙ db)
            {
                Matrix<ty> tmp_v = this->_bv;
                tmp_v.template scalar_mul<use_blas>(this->_rms, execPolicy{}); // β2*v

                Matrix<ty> db_sq = db;                          // copy
                db_sq.hadamard_mul(db, execPolicy{});           // db ⊙ db
                db_sq.template scalar_mul<use_blas>((1 - this->_rms), execPolicy{}); // (1-β2)*(db^2)

                this->_bv = tmp_v + db_sq;
            }

            // m_hat = m / (1 - β1^t)
            Matrix<ty> m_hat_b = _bm;
            m_hat_b.template scalar_mul<use_blas>(1.0 / (1 - std::pow(this->_momentum, this->_time)), execPolicy{});

            // v_hat = v / (1 - β2^t)
            Matrix<ty> v_hat_b = _bv;
            v_hat_b.template scalar_mul<use_blas>(1.0 / (1 - std::pow(this->_rms, this->_time)), execPolicy{});

            // update = m_hat / (sqrt(v_hat) + ε)
            Matrix<ty> updateB = m_hat_b;

            // divide by sqrt(v_hat) + ε
            {
                Matrix<ty> denom = v_hat_b;
                denom.apply([this](ty x){ return static_cast<ty>(1) / (std::sqrt(x) + 1e-8); }, execPolicy{});
                updateB.hadamard_mul(denom, execPolicy{});
            }

            updateB.template scalar_mul<use_blas>(this->_learning_rate, execPolicy{});
            _b.template sub<use_blas>(updateB, execPolicy{});
        }
        catch(const std::exception& e){
            std::cerr << "Error in Affine backward: " << e.what() << std::endl;
            throw;
        }
    } 
};

#endif