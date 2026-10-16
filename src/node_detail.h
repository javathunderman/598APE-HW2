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
inline void evaluate_node(const node &n, const float *data, const uint64_t stride,
                          std::valarray<float> &in0, std::valarray<float> &in1, std::valarray<float> &res) {
  if (n.t == node::type::constant) {
    res = n.u.val;
  } else if (n.t == node::type::variable) {
    res = std::valarray<float>(data + stride * n.u.fid, stride);
  } else {
    // note: keep the case statements in alphabetical order under each category
    // of operators.
    switch (n.t) {
    // binary operators
    case node::type::add: {
      res = in0 + in1;
      break;
    }
    case node::type::atan2: {
      res = std::atan2(in0, in1);
      break;
    }
    case node::type::div: {
      res = 1.0f;
      for (unsigned int i = 0; i < stride; ++i) {
        if (fabsf(in1[i]) >= MIN_VAL) res[i] = in0[i]/in1[i];
      }
      break;
    }
    case node::type::fdim: {
      for (unsigned int i = 0; i < stride; ++i)
        res[i] = fdimf(in0[i], in1[i]);
      break;
    }
    case node::type::gt: {
      res = 0.0f;
      for (unsigned int i = 0; i < stride; ++i) {
        if (in0[i] > in1[i]) res[i] = 1.0f;
      }
      break;
    }
    case node::type::max: {
      for (unsigned int i = 0; i < stride; ++i) {
        if (in0[i] >= in1[i]) res[i] = in0[i];
        else res[i] = in1[i];
      }
      break;
    }
    case node::type::min: {
      for (unsigned int i = 0; i < stride; ++i) {
        if (in0[i] <= in1[i]) res[i] = in0[i];
        else res[i] = in1[i];
      }
      break;
    } 
    case node::type::mul: {
      res = in0 * in1;
      break;
    } 
    case node::type::pow: {
      res = std::pow(in0, in1);
      break;
    }
    case node::type::sub: {
      res = in0 - in1;
      break;
    }
    // unary operators
    case node::type::abs: {
      res = std::abs(in0);
      break;
    }
    case node::type::acos: {
      res = std::acos(in0);
      break;
    }
    case node::type::acosh: {
      for (unsigned int i = 0; i < stride; ++i)
        res[i] = acoshf(in0[i]);
      break;
    }
    case node::type::asin: {
      res = std::asin(in0);
      break;
    }
    case node::type::asinh: {
      for (unsigned int i = 0; i < stride; ++i)
        res[i] = asinhf(in0[i]);
      break;
    }
    case node::type::atan: {
      res = std::atan(in0);
      break;
    }
    case node::type::atanh: {
      for (unsigned int i = 0; i < stride; ++i)
        res[i] = atanhf(in0[i]);
      break;
    }
    case node::type::cbrt: {
      for (unsigned int i = 0; i < stride; ++i)
        res[i] = cbrtf(in0[i]);
      break;
    }
    case node::type::cos: {
      res = std::cos(in0);
      break;
    }
    case node::type::cosh: {
      res = std::cosh(in0);
      break;
    }
    case node::type::cube: {
      res = in0 * in0 * in0;
      break;
    }
    case node::type::exp: {
      res = std::exp(in0);
      break;
    }
    case node::type::inv: {
      for (unsigned int i = 0; i < stride; ++i) {
        if (fabsf(in0[i]) < MIN_VAL) res[i] = 0.f;
        else res[i] = 1.f/(in0[i]);
      }
      break;
    }
    case node::type::log: {
      res = std::abs(in0);
      for (unsigned int i = 0; i < stride; ++i) {
        if (res[i] < MIN_VAL) res[i] = 0.f;
        else res[i] = logf(res[i]);
      }
      break;
    }
    case node::type::neg: {
      res = -in0;
      break;
    }
    case node::type::rcbrt: {
      for (unsigned int i = 0; i < stride; ++i)
        res[i] = static_cast<float>(1.0) / cbrtf(in0[i]);
      break;
    }
    case node::type::rsqrt: {
      for (unsigned int i = 0; i < stride; ++i)
        res[i] = static_cast<float>(1.0) / sqrtf(fabsf(in0[i]));
      break;
    }
    case node::type::sin: {
      res = std::sin(in0);
      break;
    }
    case node::type::sinh: {
      res = std::sinh(in0);
      break;
    }
    case node::type::sq: {
      res = in0 * in0;
      break;
    }
    case node::type::sqrt: {
      res = std::sqrt(std::abs(in0));
      break;
    }
    case node::type::tan: {
      res = std::tan(in0);
      break;
    }
    case node::type::tanh: {
      res = std::tanh(in0);
      break;
    }
    // shouldn't reach here!
    default:
      res = 0.0f;
      break;
    };
  }
}

} // namespace detail
} // namespace genetic
