/*
 * File: native/include/bt/ops.h
 * Purpose: Declares scalar binary operation functors used by tensor kernels.
 */

#pragma once

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

} /* namespace bt::ops */
