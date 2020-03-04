#pragma once

#include <iostream>
#include <tuple>
#include <functional>

#include "eigen/Eigen/Core"
#include "eigen/Eigen/../unsupported/Eigen/CXX11/Tensor"
using Eigen::Sizes;
template<int... ints>
using Tensor = Eigen::TensorFixedSize<float,Sizes<ints...>>;

template<int ... ints>
struct Shape {
    static constexpr std::array<int,sizeof...(ints)> shape{ints...};
    using tensor = Tensor<ints...>;
    template<int ... I>
    friend std::ostream& operator<<(std::ostream& os, const Shape<I...>& shape);
};

template<int ... ints>
std::ostream& operator<<(std::ostream& os, const Shape<ints...>& s) {
    os << "Shape<";
    for (int i = 0; int dim : s.shape) {
      if (i++) os << ",";
      os << dim;
    }
    os << ">";
    return os;
}

// #define NDEBUG
// // http://eigen.tuxfamily.org/index.php?title=FAQ#How_do_I_get_good_performance.3F
// #define EIGEN_RUNTIME_NO_MALLOC
// Eigen::internal::set_is_malloc_allowed(false);

template<class F> auto comp(F f) {
   return [f](const auto& x) { return f.forward(x); };
}
template<class F, class G, class... Funcs> auto comp(F f, G g, Funcs... funcs) {
   return [f,g,funcs...](const auto& x) { return f.forward(comp(g, funcs...)(x)); };
}

template<typename... Args> struct TypeStack {
    template<typename T> using push = TypeStack<T, Args...>;
};

template<typename vec, typename ishape, typename... Args>
struct recurse {
    using type = vec;
    using finaloshape = ishape;
};
template<typename vec, typename ishape, typename layerType, typename... Args>
struct recurse<vec, ishape, layerType, Args...> {
private:
    using oshape = typename layerType::template oshape<ishape>;
    using layerFullType = typename layerType::template type<ishape, oshape>;
    using vec_ = typename vec::template push<layerFullType>;
    using rec = recurse<vec_, oshape, Args...>;

public:
    using type = typename rec::type;
    using finaloshape = typename rec::finaloshape;
};

template<typename... Args> class NetworkReal {};
template<typename is, typename os, typename... Args> class NetworkReal<is, os, TypeStack<Args...>> {
    std::tuple<Args...> layers{Args()...};
    NetworkReal(NetworkReal&) = delete;
    NetworkReal(const NetworkReal&);
    NetworkReal& operator=(const NetworkReal&);
public:
    NetworkReal() = default;

    // std::function<typename os::tensor(const typename is::tensor&)>
    //f{std::apply([](auto&& ... layer){ return comp(layer...); }, layers)};

    typename os::tensor f(const typename is::tensor& x) {

    // template<int i=0, class T>
    // auto f(T&& x) {
    //     if constexpr(i == sizeof...(Args))
    //       return x;
    //     else
    //       return std::get<i>(layers).forward(f<i+1>(x));

        if constexpr (sizeof...(Args) == 2) {
          return std::get<0>(layers).forward(
                 std::get<1>(layers).forward(x));
        } else if constexpr (sizeof...(Args) == 3) {
          return std::get<0>(layers).forward(
                 std::get<1>(layers).forward(
                 std::get<2>(layers).forward(x)));
        } else if constexpr (sizeof...(Args) == 4) {
          return std::get<0>(layers).forward(
                 std::get<1>(layers).forward(
                 std::get<2>(layers).forward(
                 std::get<3>(layers).forward(x))));
        } else if constexpr (sizeof...(Args) == 5) {
          return std::get<0>(layers).forward(
                 std::get<1>(layers).forward(
                 std::get<2>(layers).forward(
                 std::get<3>(layers).forward(
                 std::get<4>(layers).forward(x)))));
        } else if constexpr (sizeof...(Args) == 6) {
          return std::get<0>(layers).forward(
                 std::get<1>(layers).forward(
                 std::get<2>(layers).forward(
                 std::get<3>(layers).forward(
                 std::get<4>(layers).forward(
                 std::get<5>(layers).forward(x))))));
        } else if constexpr (sizeof...(Args) == 7) {
          return std::get<0>(layers).forward(
                 std::get<1>(layers).forward(
                 std::get<2>(layers).forward(
                 std::get<3>(layers).forward(
                 std::get<4>(layers).forward(
                 std::get<5>(layers).forward(
                 std::get<6>(layers).forward(x)))))));
        }

        // comp(std::get<0>(layers),
        // std::get<1>(layers),
        // std::get<2>(layers))(x);
    }
};

template<typename is, typename... Args>
using Network = NetworkReal<is,
    typename recurse<TypeStack<>, is, Args...>::finaloshape,
    typename recurse<TypeStack<>, is, Args...>::type>;
