#pragma once

#include <array>
#include <utility>

template<int fh, int fw, typename is, typename os>
class ConvReLUReal {
    Tensor<fh,fw,is::shape[3],os::shape[3]> F;
public:
    ConvReLUReal() { F.setConstant(1.f/(fh*fw*is::shape[3])); }
    template<typename T> auto forward(const T& x) {
        std::cout << "ConvReLU<" << fh << "," << fw << "," << is() << "," << os() << ">" << std::endl;
        //https://stackoverflow.com/questions/58788433/how-to-use-eigentensorconvolve-with-multiple-kernels
        typename os::tensor y;
        for (int i = 0; i < os::shape[3]; ++i)
            y.chip(i, 3) = x.convolve(F.chip(i, 3), std::array<int,3>{1,2,3}).chip(0,3);
        return y;
    }
};
template<int od, int fh=3, int fw=3>
struct ConvReLU {
    template<typename is, typename os>
    using type = ConvReLUReal<fh,fw,is,os>;
    template<typename is>
    using oshape = Shape<is::shape[0],is::shape[1]-fh+1,is::shape[2]-fw+1,od>;
};

template<typename is, typename os>
struct FlattenReal {
template<typename T> auto forward(const T& x) const {
        std::cout << "Flatten<" << is() << "," << os() << ">" << std::endl;
        return x.reshape(os::shape);
    }
};
struct Flatten {
    template<typename is, typename os> using type = FlattenReal<is,os>;
    template<typename is> using oshape = Shape<is::shape[0], is::shape[1]*is::shape[2]*is::shape[3]>;
};

template<typename is, typename os>
class DenseReal {
    Tensor<is::shape[1], os::shape[1]> W;
public:
    DenseReal() { W.setConstant(1.f/is::shape[1]); }
    template<typename T> auto forward(const T& x) const {
        std::cout << "Dense<" << is() << "," << os() << ">" << std::endl;
        constexpr std::array<std::pair<int,int>,1> product_dims = {{{1,0}}};
        return x.contract(W, product_dims);
    }
};
template<int ocols>
struct Dense {
    template<typename is, typename os> using type = DenseReal<is,os>;
    template<typename is> using oshape = Shape<is::shape[0],ocols>;
};

// Computes the softmax of a batch of row vectors
template<typename is, typename os>
struct SoftMaxReal {
    template<typename T> auto forward(const T& x) const {
      auto y = x.exp().eval();
      constexpr auto whichDim = std::array<int,1>{1};
      auto sum = y.sum(whichDim).reshape(std::array<int,2>{is::shape[0],1}); // empty dim
      constexpr auto replicateDimTimes = std::array<int,2>{1,is::shape[1]};
      auto resize = sum.broadcast(replicateDimTimes);
      return y / resize;
    }
};
struct SoftMax {
    template<typename is, typename os> using type = SoftMaxReal<is,os>;
    template<typename is> using oshape = Shape<is::shape[0],is::shape[1]>;
};
