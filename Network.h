#pragma once

#include <tuple>
#include <functional>
#include <array>
#include <utility>

// TODO compile time random: mklimenko.github.io/english/2018/06/04/constexpr-random/

// These two lines are VERY different in my code.
// If every layer uses the first line for its forward declaration, then the stack gets messed up.
//
// 1:  auto inline forward(const T& x) const;
//
// 2:  inline typename os::tensor forward(const typename is::tensor& x) const;
//
// The problem here is mentioned in Eigen's documentation.
// Search for "eigen3 common pitfalls" and read the whole article.

// I think the constinit keywords are not neccessary

// layerid forces every instantiation of a layer to be a different type

#include "eigen/Eigen/Core"
#include "eigen/Eigen/../unsupported/Eigen/CXX11/Tensor"
using Eigen::Sizes;
template<int... ints>
using Tensor = Eigen::TensorFixedSize<double,Sizes<ints...>>;

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

template<typename... Args> struct TypeStack {
    template<typename T> using push = TypeStack<T, Args...>;
};

template<int i, typename vec, typename ishape, typename... Args>
struct recurse {
    using type = vec;
    using finaloshape = ishape;
};
template<int i, typename vec, typename ishape, typename layerType, typename... Args>
struct recurse<i, vec, ishape, layerType, Args...> {
  private:
    using oshape = typename layerType::template oshape<ishape>;
    using layerFullType = typename layerType::template type<ishape, oshape, i>;
    using vec_ = typename vec::template push<layerFullType>;
    using rec = recurse<i+1, vec_, oshape, Args...>;

  public:
    using type = typename rec::type;
    using finaloshape = typename rec::finaloshape;
};

template<typename... Args> class NetworkReal {};
template<typename is_, typename os_, typename... Args> class NetworkReal<is_, os_, TypeStack<Args...>> {
    NetworkReal(NetworkReal&) = delete;
    NetworkReal(const NetworkReal&);
    NetworkReal& operator=(const NetworkReal&);
  public:
    constinit static inline std::tuple<Args...> layers{Args()...};
    using is = is_;
    using os = os_;
    consteval NetworkReal() = default;
    void init() {
        std::apply([](auto&& ... layer){ (..., layer.init()); }, layers);
    }
    template<int i=0, class T>
    inline auto forward(const T& x) {
       if constexpr(i >= sizeof...(Args))
         return x;
       else
         return std::get<i>(layers).forward(forward<i+1>(x));
    }
    template<int i=sizeof...(Args)-1, class T>
    inline auto backward(const T& x) {
       if constexpr(i < 0)
         return x;
       else
         return std::get<i>(layers).backward(backward<i-1>(x));
    }
};

template<typename is, typename... Args>
using Network = NetworkReal<is,
    typename recurse<0, TypeStack<>, is, Args...>::finaloshape,
    typename recurse<0, TypeStack<>, is, Args...>::type>;
