#pragma once

#include <tuple>
#include <functional>
#include <array>
#include <utility>

// TODO static polymorphism

namespace Layers {

using std::array;
using std::pair;
using std::cout;
using std::endl;

template<int fh, int fw, typename is, typename os, int layerid>
struct ConvReal {
    constinit static inline Tensor<fh,fw,is::shape[3],os::shape[3]> F;
    constinit static inline Tensor<fh,fw,is::shape[3],os::shape[3]> dF;
    constinit static inline typename is::tensor x_; // TODO bad
    consteval ConvReal() = default;
    void init() {
        F.setRandom();
        F *= F.constant(sqrt(1.f/(fh*fw*is::shape[3]*os::shape[3])));
    }
    inline typename os::tensor forward(const typename is::tensor& x) const {
        x_ = x;
        cout << "Conv " << fh << "x" << fw << ": " << is() << " -> " << os() << endl;
        //https://stackoverflow.com/questions/58788433/how-to-use-eigentensorconvolve-with-multiple-kernels
        typename os::tensor z;
        for (int i = 0; i < os::shape[3]; ++i)
            z.chip(i, 3) = x.convolve(F.chip(i, 3), array<int,3>{1,2,3}).chip(0,3);
        return z;
    }
    inline typename is::tensor backward(const typename os::tensor& dz) {
        cout << "dConv " << fh << "x" << fw << ": " << os() << " -> " << is() << endl;
        // x  = b ih iw id
        // dz = b oh ow od
        // dF = fh fw id od

        // x  = 2 3 4 5
        // dz = 2 1 2 8
        // dF = 3 3 5 8

        // Just like in the Dense layer, we loose the batch dimension at this step.
        // The first 'row (=depth)' of x 'multiplied (=convolved)' with the first row of dz?
        // so x.conv(F) (=x*F) is a sum over x_i*F_i where ...?
        for (int i = 0; i < os::shape[3]; ++i)
            dF.chip(i, 3) = x_.convolve(dz.chip(i, 3), array<int,3>{0,1,2}).chip(0,0);

        typename is::tensor dx;

        constexpr array<pair<int,int>,4> paddings{{{0,0}, {fh-1,fh-1}, {fw-1,fw-1}, {0,0}}};

        // Tensor<is::shape[0],is::shape[1]+fh-1,is::shape[2]+fw-1,os::shape[3]> padded = dz.pad(paddings);
        auto padded = dz.pad(paddings);
        for (int i = 0; i < is::shape[3]; ++i) {
            auto frev = F.chip(i, 2).reverse(array<bool,3>{true, true, false});
            // Tensor<is::shape[0],is::shape[1],is::shape[2]> padconv = padded.convolve(frev,array<int,3>{1,2,3}).chip(0,3);
            auto padconv = padded.convolve(frev,array<int,3>{1,2,3}).chip(0,3);
            dx.chip(i, 3) = padconv;
        }

        return dx;
    }
};
template<int od, int fh=3, int fw=3>
struct Conv {
    template<typename is, typename os, int layerid>
    using type = ConvReal<fh,fw,is,os,layerid>;
    template<typename is>
    using oshape = Shape<is::shape[0],is::shape[1]-fh+1,is::shape[2]-fw+1,od>;
};

template<typename is, typename os>
struct FlattenReal {
    consteval FlattenReal() = default;
    void init() {};
    inline typename os::tensor forward(const typename is::tensor& x) const {
        cout << "Flatten: " << is() << " -> " << os() << endl;
        return x.reshape(os::shape);
    }
    inline typename is::tensor backward(const typename os::tensor& dz) {
        cout << "dFlatten: " << os() << " -> " << is() << endl;
        return dz.reshape(is::shape);
    }
};
struct Flatten {
    template<typename is, typename os, int layerid> using type = FlattenReal<is,os>;
    template<typename is> using oshape = Shape<is::shape[0], is::shape[1]*is::shape[2]*is::shape[3]>;
};

template<typename is, typename os, int layerid> struct DenseReal {
    constinit static inline Tensor<is::shape[1], os::shape[1]> W;
    constinit static inline Tensor<is::shape[1], os::shape[1]> dW;
    constinit static inline typename is::tensor x_;
    consteval DenseReal() = default;
    void init() {
        W.setRandom();
        W *= W.constant(sqrt(1.f/(os::shape[1]*is::shape[1])));
    }
    inline typename os::tensor forward(const typename is::tensor& x) const {
        cout << "Dense: " << is() << " -> " << os() << endl;
        x_ = x;
        constexpr array<pair<int,int>,1> product_dims = {{{1,0}}};
        return x.contract(W, product_dims); // x*W => z
    }
    inline typename is::tensor backward(const typename os::tensor& dz) {
        cout << "dDense: " << os() << " -> " << is() << endl;
        // The matrix product AB can be written as a sum of outer products of columns of A and rows of B: sum_{1,N}((A^T_i)^T*B_i)
        // dW = sum_{1,N}(x_i^T*dz_i) where x_i^T*dz_i is the gradient w.r.t W for the ith batch element
        constexpr array<pair<int,int>,1> dims1 = {{{0,0}}};
        dW = x_.contract(dz, dims1); // x^T*dz
        constexpr array<pair<int,int>,1> dims2 = {{{1,1}}};
        return dz.contract(W, dims2); // dz*W^T = dx
    }
};
template<int ocols> struct Dense {
    template<typename is, typename os, int layerid> using type = DenseReal<is,os,layerid>;
    template<typename is> using oshape = Shape<is::shape[0],ocols>;
};

// Computes the softmax of a batch of row vectors
template<typename is, typename os, int layerid>
struct SoftMaxReal {
    consteval SoftMaxReal() = default;
    void init() {};
    inline typename os::tensor forward(const typename is::tensor& x) const {
      auto z = x.exp().eval();
      constexpr auto whichDim = array<int,1>{1};
      auto sum = z.sum(whichDim).reshape(array<int,2>{is::shape[0],1}); // empty dim
      constexpr auto replicateDimTimes = array<int,2>{1,is::shape[1]};
      auto resize = sum.eval().broadcast(replicateDimTimes);
      auto a = z / resize;
      return a;
    }
};
struct SoftMax {
    template<typename is, typename os, int layerid> using type = SoftMaxReal<is,os,layerid>;
    template<typename is> using oshape = Shape<is::shape[0],is::shape[1]>;
};

} // end namespace Layers