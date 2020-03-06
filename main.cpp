#include <iostream>

#include "Network.h"
#include "Layers.h"
using Layers::Conv;
using Layers::Flatten;
using Layers::Dense;

using NetworkT = Network<Shape<4, 28, 28, 2>,
                         Conv<8,7,7>,
                         Conv<4,5,5>,
                         Conv<1,5,5>,
                         Conv<1,5,5>,
                         Conv<1>,
                         Flatten, // TODO is there a way to let Dense take a 1,10,1,1 mat?
                         Dense<4>,
                         Dense<4>,
                         Dense<4>,
                         Dense<4>>;
NetworkT network;

NetworkT::is::tensor X;
NetworkT::is::tensor A0;
NetworkT::is::tensor A1;

int main() {
    std::cout.precision(14);
    network.init();

    // Just evaluate
    // X.setConstant(.5f);
    // // const NetworkT::os::tensor Y = network.forward(X);
    // const auto Y = network.forward(X);
    // std::cout << Y << std::endl << std::endl;
    // NetworkT::os::tensor O;
    // O.setConstant(.5f);
    // const NetworkT::is::tensor D = network.backward(O);
    // std::cout << D << std::endl;

    // Do some grad checks
    X.setRandom();
    X = (2.*X-X.constant(.5)).eval();
    // A0=X;
    // A1=X;
    // A0(2,12,12,0) -= .0001f;
    // A1(2,12,12,0) += .0001f;
    // const Eigen::Tensor<double,0> y0 = network.forward(A0).sum();
    // std::cout << y0 << std::endl;
    // const Eigen::Tensor<double,0> y1 = network.forward(A1).sum();
    // std::cout << y1 << std::endl;
    // const double diffquo = (y1.coeff(0)-y0.coeff(0)) / .0002f;
    // std::cout << diffquo << std::endl;
    //
    // NetworkT::os::tensor sumgrad;
    // sumgrad.setConstant(1.f);
    // const Eigen::Tensor<double,0> y2 = network.forward(X).sum();
    // std::cout << y2 << std::endl;
    // const Eigen::Tensor<double,4> inputgrad = network.backward(sumgrad);
    // std::cout << inputgrad(2,12,12,0) << std::endl;

    // // Check the conv layer
    auto& F = std::get<9>(network.layers).F;
    auto& dF = std::get<9>(network.layers).dF;
    F(4,4,0,5) -= .0001f;
    const Eigen::Tensor<double,0> y0 = network.forward(X).sum();
    std::cout << y0 << std::endl;
    F(4,4,0,5) += .0002f;
    const Eigen::Tensor<double,0> y1 = network.forward(X).sum();
    std::cout << y1 << std::endl;
    const double diffquo = (y1.coeff(0)-y0.coeff(0)) / .0002f;
    std::cout << diffquo << std::endl;
    F(4,4,0,5) -= .0001f;
    NetworkT::os::tensor sumgrad;
    sumgrad.setConstant(1.f);
    const Eigen::Tensor<double,0> y2 = network.forward(X).sum();
    std::cout << y2 << std::endl;
    network.backward(sumgrad);
    std::cout << dF(4,4,0,5) << std::endl;

    // // Check the dense layer
    // auto& W = std::get<2>(network.layers).W;
    // auto& dW = std::get<2>(network.layers).dW;
    // W(2,2) -= .0001f;
    // const Eigen::Tensor<double,0> y0 = network.forward(X).sum();
    // std::cout << y0 << std::endl;
    // W(2,2) += .0002f;
    // const Eigen::Tensor<double,0> y1 = network.forward(X).sum();
    // std::cout << y1 << std::endl;
    // const double diffquo = (y1.coeff(0)-y0.coeff(0)) / .0002f;
    // std::cout << diffquo << std::endl;
    // W(2,2) -= .0001f;
    // NetworkT::os::tensor sumgrad;
    // sumgrad.setConstant(1.f);
    // const Eigen::Tensor<double,0> y2 = network.forward(X).sum();
    // std::cout << y2 << std::endl;
    // network.backward(sumgrad);
    // std::cout << dW(2,2) << std::endl;

    std::cout.precision(4);
    std::cout << X.chip(0,0).chip(0,2) << std::endl;
}
