/*
 * File: native/include/bt/ops.h
 * Purpose: Declares scalar operation functors used by tensor kernels.
 */

#pragma once

#include <cmath>

/*
 * Namespace: bt::ops
 * Purpose: Lightweight operation objects for elementwise execution.
 */
namespace bt::ops {

/*
 * Functor: Add
 * Purpose: Computes x + y.
 */
struct Add {
  float operator()(float x, float y) const noexcept { return x + y; }
};

/*
 * Functor: Sub
 * Purpose: Computes x - y.
 */
struct Sub {
  float operator()(float x, float y) const noexcept { return x - y; }
};

/*
 * Functor: Mul
 * Purpose: Computes x * y.
 */
struct Mul {
  float operator()(float x, float y) const noexcept { return x * y; }
};

/*
 * Functor: Div
 * Purpose: Computes x / y.
 */
struct Div {
  float operator()(float x, float y) const noexcept { return x / y; }
};

/*
 * Functor: Eq
 * Purpose: Computes x == y.
 */
struct Eq {
  template <typename T> bool operator()(T x, T y) const noexcept { return x == y; }
};

/*
 * Functor: Ne
 * Purpose: Computes x != y.
 */
struct Ne {
  template <typename T> bool operator()(T x, T y) const noexcept { return x != y; }
};

/*
 * Functor: Lt
 * Purpose: Computes x < y.
 */
struct Lt {
  template <typename T> bool operator()(T x, T y) const noexcept { return x < y; }
};

/*
 * Functor: Le
 * Purpose: Computes x <= y.
 */
struct Le {
  template <typename T> bool operator()(T x, T y) const noexcept { return x <= y; }
};

/*
 * Functor: Gt
 * Purpose: Computes x > y.
 */
struct Gt {
  template <typename T> bool operator()(T x, T y) const noexcept { return x > y; }
};

/*
 * Functor: Ge
 * Purpose: Computes x >= y.
 */
struct Ge {
  template <typename T> bool operator()(T x, T y) const noexcept { return x >= y; }
};

/*
 * Functor: Exp
 * Purpose: Computes exp(x).
 */
struct Exp {
  float operator()(float x) const noexcept { return std::exp(x); }
};

/*
 * Functor: Log
 * Purpose: Computes log(x).
 */
struct Log {
  float operator()(float x) const noexcept { return std::log(x); }
};

/*
 * Functor: Tanh
 * Purpose: Computes tanh(x).
 */
struct Tanh {
  float operator()(float x) const noexcept { return std::tanh(x); }
};

/*
 * Functor: Sigmoid
 * Purpose: Computes the logistic sigmoid using a branch-stable formula.
 */
struct Sigmoid {
  float operator()(float x) const noexcept {
    if (x >= 0.0f) {
      const float exp_neg_x = std::exp(-x);
      return 1.0f / (1.0f + exp_neg_x);
    }

    const float exp_x = std::exp(x);
    return exp_x / (1.0f + exp_x);
  }
};

} /* namespace bt::ops */
