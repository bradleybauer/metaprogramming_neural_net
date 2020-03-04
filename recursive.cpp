#include <iostream>
#include <array>
using std::array;

#include "Network.h"
#include "Layers.h"

// Network<Shape<2, 28, 28, 1>,
//         ConvReLU<8>,
//         ConvReLU<4>,
//         ConvReLU<2>,
//         ConvReLU<1>,
//         Flatten,
//         Dense<10>> network;

Network<Shape<1, 8, 8, 2>,
  ConvReLU<2>,
  ConvReLU<2>,
  Flatten,
  Dense<10>,
  SoftMax> network;

// Network<Shape<1, 4, 4, 2>,
//   Flatten,
//   Dense<10>> network;

int main() {
    // Tensor<2, 28, 28, 1> X;
    Tensor<1, 8, 8, 2> X;
    X.setConstant(.5f);
    const auto Y = network.f(X);
    std::cout << Y << std::endl;
}
