#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <node.h>
#include <valarray>

namespace genetic {
namespace detail {

static constexpr float MIN_VAL = 0.001f;

inline bool is_terminal(node::type t) {
  return t == node::type::variable || t == node::type::constant;
}

inline bool is_nonterminal(node::type t) { return !is_terminal(t); }

inline int arity(node::type t) {
  if (node::type::unary_begin <= t && t <= node::type::unary_end) {
    return 1;
  }
  if (node::type::binary_begin <= t && t <= node::type::binary_end) {
    return 2;
  }
  return 0;
}

// `data` assumed to be stored in col-major format
inline std::valarray<float> evaluate_node(const node &n, const float *data, const uint64_t stride,
                          std::valarray<float> in0, std::valarray<float> in1) {
  if (n.t == node::type::constant) {
    return std::valarray<float>(n.u.val, stride);
  } else if (n.t == node::type::variable) {
    return std::valarray<float>(data + stride * n.u.fid, stride);
  } else {
    // note: keep the case statements in alphabetical order under each category
    // of operators.
    switch (n.t) {
    // binary operators
    case node::type::add:
      return in0 + in1;
    case node::type::atan2:
      return std::atan2(in0, in1);
    case node::type::div: {
      auto arr = std::valarray<float>(1.0f, stride);
      for (unsigned int i = 0; i < stride; ++i) {
        if (fabsf(in1[i]) >= MIN_VAL) arr[i] = in0[i]/in1[i];
      }
      return arr;
    }
    case node::type::fdim: {
      auto arr = std::valarray<float>(stride);
      for (unsigned int i = 0; i < stride; ++i)
        arr[i] = fdimf(in0[i], in1[i]);
      return arr;
    }
    case node::type::gt: {
      auto arr = std::valarray<float>(0.0f, stride);
      for (unsigned int i = 0; i < stride; ++i) {
        if (in0[i] > in1[i]) arr[i] = 1.0f;
      }
      return arr;
    }
    case node::type::max: {
      auto arr = std::valarray<float>(stride);
      for (unsigned int i = 0; i < stride; ++i) {
        if (in0[i] >= in1[i]) arr[i] = in0[i];
        else arr[i] = in1[i];
      }
      return arr;
    }
    case node::type::min: {
      auto arr = std::valarray<float>(stride);
      for (unsigned int i = 0; i < stride; ++i) {
        if (in0[i] <= in1[i]) arr[i] = in0[i];
        else arr[i] = in1[i];
      }
      return arr;
    }
    case node::type::mul:
      return in0 * in1;
    case node::type::pow:
      return std::pow(in0, in1);
    case node::type::sub:
      return in0 - in1;
    // unary operators
    case node::type::abs:
      return std::abs(in0);
    case node::type::acos:
      return std::acos(in0);
    case node::type::acosh: {
      auto arr = std::valarray<float>(stride);
      for (unsigned int i = 0; i < stride; ++i)
        arr[i] = acoshf(in0[i]);
      return arr;
    }
    case node::type::asin:
      return std::asin(in0);
    case node::type::asinh: {
      auto arr = std::valarray<float>(stride);
      for (unsigned int i = 0; i < stride; ++i)
        arr[i] = asinhf(in0[i]);
      return arr;
    }
    case node::type::atan:
      return std::atan(in0);
    case node::type::atanh: {
      auto arr = std::valarray<float>(stride);
      for (unsigned int i = 0; i < stride; ++i)
        arr[i] = atanhf(in0[i]);
      return arr;
    }
    case node::type::cbrt: {
      auto arr = std::valarray<float>(stride);
      for (unsigned int i = 0; i < stride; ++i)
        arr[i] = cbrtf(in0[i]);
      return arr;
    }
    case node::type::cos:
      return std::cos(in0);
    case node::type::cosh:
      return std::cosh(in0);
    case node::type::cube:
      return in0 * in0 * in0;
    case node::type::exp:
      return std::exp(in0);
    case node::type::inv: {
      auto arr = std::valarray<float>(stride);
      for (unsigned int i = 0; i < stride; ++i) {
        if (fabsf(in0[i]) < MIN_VAL) arr[i] = 0.f;
        else arr[i] = 1.f/(in0[i]);
      }
      return arr;
    }
    case node::type::log: {
      std::valarray<float> arr = std::abs(in0);
      for (unsigned int i = 0; i < stride; ++i) {
        if (arr[i] < MIN_VAL) arr[i] = 0.f;
        else arr[i] = logf(arr[i]);
      }
      return arr;
    }
    case node::type::neg:
      return -in0;
    case node::type::rcbrt: {
      auto arr = std::valarray<float>(stride);
      for (unsigned int i = 0; i < stride; ++i)
        arr[i] = static_cast<float>(1.0) / cbrtf(in0[i]);
      return arr;
    }
    case node::type::rsqrt: {
      auto arr = std::valarray<float>(stride);
      for (unsigned int i = 0; i < stride; ++i)
        arr[i] = static_cast<float>(1.0) / sqrtf(fabsf(in0[i]));
      return arr;
    }
    case node::type::sin:
      return std::sin(in0);
    case node::type::sinh:
      return std::sinh(in0);
    case node::type::sq:
      return in0 * in0;
    case node::type::sqrt:
      return std::sqrt(std::abs(in0));
    case node::type::tan:
      return std::tan(in0);
    case node::type::tanh:
      return std::tanh(in0);
    // shouldn't reach here!
    default:
      return std::valarray<float>(0.0f, stride);
    };
  }
}

} // namespace detail
} // namespace genetic
