#include <iostream>

#include "Network.h"
#include "Layers.h"

using NetworkT = Network<Shape<4, 28, 28, 3>,
                         Conv<8,7,7>,
                         Conv<4,5,5>,
                         Conv<2>,
                         Conv<1>,
                         Flatten, // TODO is there a way to let Dense take a 1,10,1,1 mat?
                         Dense<4>>;
NetworkT network;

int main() {
    network.init();

    NetworkT::is::tensor X;
    // X.setConstant(.5f);
    // // const NetworkT::os::tensor Y = network.forward(X);
    // const auto Y = network.forward(X);
    // std::cout << Y << std::endl << std::endl;

    // NetworkT::os::tensor O;
    // O.setConstant(.5f);
    // const NetworkT::is::tensor D = network.backward(O);
    // std::cout << D << std::endl;

    X.setRandom();
    NetworkT::is::tensor A0 = X;
    NetworkT::is::tensor A1 = X;
    A0(0,2,1,1) -= .00001f;
    A1(0,2,1,1) += .00001f;
    const Eigen::Tensor<double,0> y0 = network.forward(A0).sum();
    std::cout << y0 << std::endl;
    const Eigen::Tensor<double,0> y1 = network.forward(A1).sum();
    std::cout << y1 << std::endl;
    const double diffquo = (y1.coeff(0)-y0.coeff(0)) / .00002f;
    std::cout << diffquo << std::endl;

    NetworkT::os::tensor sumgrad;
    sumgrad.setConstant(1.f);
    const auto y2 = network.forward(X).sum();
    std::cout << y2 << std::endl;
    const auto inputgrad = network.backward(sumgrad);
    const auto gradient = inputgrad(0,2,1,1);
    std::cout << gradient << std::endl;
}