#pragma once

#include <tuple>
#include <functional>
#include <array>
#include <utility>

// TODO static polymorphism

template<int fh, int fw, typename is, typename os, int layerid>
class ConvReal {
    constinit static inline Tensor<fh,fw,is::shape[3],os::shape[3]> F;
    constinit static inline Tensor<fh,fw,is::shape[3],os::shape[3]> dF;
    constinit static inline typename is::tensor x_; // TODO bad
  public:
    consteval ConvReal() = default;
    void init() {
        // F.setConstant(1.f/(fh*fw*is::shape[3]));
        F.setRandom();
        F *= F.constant(sqrt(1.f/(fh*fw*is::shape[3])));
    }
    inline typename os::tensor forward(const typename is::tensor& x) const {
        x_ = x;
        std::cout << "Conv " << fh << "x" << fw << ": " << is() << " -> " << os() << std::endl;
        //https://stackoverflow.com/questions/58788433/how-to-use-eigentensorconvolve-with-multiple-kernels
        typename os::tensor y;
        for (int i = 0; i < os::shape[3]; ++i)
            y.chip(i, 3) = x.convolve(F.chip(i, 3), std::array<int,3>{1,2,3}).chip(0,3);
        return y;
    }
    inline typename is::tensor backward(const typename os::tensor& dz) {
        std::cout << "dConv " << fh << "x" << fw << ": " << os() << " -> " << is() << std::endl;
        // x  = b ih iw id
        // dz = b oh ow od
        // dF = fh fw id od

        // x  = 2 3 4 5
        // dz = 2 1 2 8
        // dF = 3 3 5 8

        constexpr auto sumDim = std::array<int,1>{0};
        for (int i = 0; i < os::shape[3]; ++i) // for each filter
            dF.chip(i, 3) = x_.convolve(dz.chip(i, 3), std::array<int,3>{0,1,2}).chip(0,0);

        typename is::tensor dx;

        constexpr std::array<std::pair<int,int>,4> paddings{{{0,0}, {fh-1,fh-1}, {fw-1,fw-1}, {0,0}}};

        // Tensor<is::shape[0],is::shape[1]+fh-1,is::shape[2]+fw-1,os::shape[3]> padded = dz.pad(paddings);
        auto padded = dz.pad(paddings);
        for (int i = 0; i < is::shape[3]; ++i) {
            // Tensor<fh,fw,os::shape[3]> frev = F.chip(i, 2).reverse(std::array<bool,3>{true, true, false});
            auto frev = F.chip(i, 2).reverse(std::array<bool,3>{true, true, false});
            // Tensor<is::shape[0],is::shape[1],is::shape[2]> padconv = padded.convolve(frev,std::array<int,3>{1,2,3}).chip(0,3);
            auto padconv = padded.convolve(frev,std::array<int,3>{1,2,3}).chip(0,3);
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
        std::cout << "Flatten: " << is() << " -> " << os() << std::endl;
        return x.reshape(os::shape);
    }
    inline typename is::tensor backward(const typename os::tensor& dz) {
        std::cout << "dFlatten: " << os() << " -> " << is() << std::endl;
        return dz.reshape(is::shape);
    }
};
struct Flatten {
    template<typename is, typename os, int layerid> using type = FlattenReal<is,os>;
    template<typename is> using oshape = Shape<is::shape[0], is::shape[1]*is::shape[2]*is::shape[3]>;
};

template<typename is, typename os, int layerid>
class DenseReal {
    constinit static inline Tensor<is::shape[1], os::shape[1]> W;
    constinit static inline Tensor<is::shape[1], os::shape[1]> dW;
    constinit static inline typename is::tensor x_; // TODO bad
  public:
    consteval DenseReal() = default;
    void init() {
        // W.setConstant(1.f/is::shape[1]);
        W.setRandom();
        W *= W.constant(sqrt(1.f/is::shape[1]));
    }
    inline typename os::tensor forward(const typename is::tensor& x) const {
        std::cout << "Dense: " << is() << " -> " << os() << std::endl;
        x_ = x;
        constexpr std::array<std::pair<int,int>,1> product_dims = {{{1,0}}};
        return x.contract(W, product_dims); // x*W => z
    }
    inline typename is::tensor backward(const typename os::tensor& dz) {
        std::cout << "dDense: " << os() << " -> " << is() << std::endl;
        // z = xW
        // c(z)
        // dcdW = dzdW * dcdz
        // dzdW = x^T
        // dW = x^T * dz
        // AB can be written as a sum of outer products of columns of A and rows of B: sum_{1,N}((A^T_i)^T*B_i)
        // dW = sum_{1,N}(x_i^T*dz_i) where x_i^T*dz_i is the gradient w.r.t W for the ith batch element
        constexpr std::array<std::pair<int,int>,1> dims1 = {{{0,0}}};
        dW = x_.contract(dz, dims1); // x^T*dz
        constexpr std::array<std::pair<int,int>,1> dims2 = {{{1,1}}};
        return dz.contract(W, dims2); // dz*W^T = dx
    }
};
template<int ocols>
struct Dense {
    template<typename is, typename os, int layerid> using type = DenseReal<is,os,layerid>;
    template<typename is> using oshape = Shape<is::shape[0],ocols>;
};

// Computes the softmax of a batch of row vectors
template<typename is, typename os, int layerid>
struct SoftMaxReal {
    consteval SoftMaxReal() = default;
    inline typename os::tensor forward(const typename is::tensor& x) const {
      auto y = x.exp().eval();
      constexpr auto whichDim = std::array<int,1>{1};
      auto sum = y.sum(whichDim).reshape(std::array<int,2>{is::shape[0],1}); // empty dim
      constexpr auto replicateDimTimes = std::array<int,2>{1,is::shape[1]};
      auto resize = sum.eval().broadcast(replicateDimTimes);
      return y / resize;
    }
    void init() {};
};
struct SoftMax {
    template<typename is, typename os, int layerid> using type = SoftMaxReal<is,os,layerid>;
    template<typename is> using oshape = Shape<is::shape[0],is::shape[1]>;
};