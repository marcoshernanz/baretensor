/*
 * File: native/src/ops.cpp
 * Purpose: Implements elementwise tensor operations and scalar variants.
 */

#include "bt/ops.h"

#include <cstdint>
#include <vector>

#include "bt/detail/broadcast.h"
#include "bt/tensor.h"

/*
 * Namespace: (anonymous)
 * Purpose: Private implementation details local to this translation unit.
 */
namespace {

/*
 * Applies a tensor-tensor operation by recursively traversing N-D shape space.
 */
template <class Op>
void recursive_apply_tt(int dim, int num_dims,
                        const std::vector<int64_t>& shape, const float* a,
                        const float* b, float* out,
                        const std::vector<int64_t>& stride_a,
                        const std::vector<int64_t>& stride_b,
                        const std::vector<int64_t>& stride_out, const Op& op) {
  if (shape[dim] == 0) return;
  if (dim == num_dims - 1) {
    for (int64_t i = 0; i < shape[dim]; ++i) {
      *out = op(*a, *b);
      a += stride_a[dim];
      b += stride_b[dim];
      out += stride_out[dim];
    }
    return;
  }

  for (int64_t i = 0; i < shape[dim]; ++i) {
    recursive_apply_tt(dim + 1, num_dims, shape, a, b, out, stride_a, stride_b,
                       stride_out, op);
    a += stride_a[dim];
    b += stride_b[dim];
    out += stride_out[dim];
  }
}

/*
 * Applies a tensor-scalar operation by recursively traversing N-D shape space.
 */
template <class Op>
void recursive_apply_ts(int dim, int num_dims,
                        const std::vector<int64_t>& shape, const float* a,
                        float s, float* out,
                        const std::vector<int64_t>& stride_a,
                        const std::vector<int64_t>& stride_out, const Op& op) {
  if (shape[dim] == 0) return;
  if (dim == num_dims - 1) {
    for (int64_t i = 0; i < shape[dim]; ++i) {
      *out = op(*a, s);
      a += stride_a[dim];
      out += stride_out[dim];
    }
    return;
  }

  for (int64_t i = 0; i < shape[dim]; ++i) {
    recursive_apply_ts(dim + 1, num_dims, shape, a, s, out, stride_a,
                       stride_out, op);
    a += stride_a[dim];
    out += stride_out[dim];
  }
}

/*
 * Executes a tensor-tensor elementwise operation with broadcasting support.
 */
template <class Op>
bt::Tensor binary_tt(const bt::Tensor& a, const bt::Tensor& b, Op op) {
  const std::vector<int64_t> out_shape =
      bt::detail::infer_broadcast_shape(a.shape, b.shape);
  bt::Tensor out(out_shape);

  const int64_t n = out.numel();
  if (n == 0) return out;

  const bool no_broadcast = (a.shape == out_shape) && (b.shape == out_shape);
  if (no_broadcast && a.is_contiguous() && b.is_contiguous() &&
      out.is_contiguous()) {
    const float* a_ptr = a.data_ptr();
    const float* b_ptr = b.data_ptr();
    float* out_ptr = out.data_ptr();
    for (int64_t i = 0; i < n; ++i) {
      out_ptr[i] = op(a_ptr[i], b_ptr[i]);
    }
    return out;
  }

  const int num_dims = out.dim();
  if (num_dims == 0) {
    *out.data_ptr() = op(*a.data_ptr(), *b.data_ptr());
    return out;
  }

  const std::vector<int64_t> stride_a =
      bt::detail::aligned_broadcast_strides(a.shape, a.strides, out_shape);
  const std::vector<int64_t> stride_b =
      bt::detail::aligned_broadcast_strides(b.shape, b.strides, out_shape);

  recursive_apply_tt(0, num_dims, out_shape, a.data_ptr(), b.data_ptr(),
                     out.data_ptr(), stride_a, stride_b, out.strides, op);
  return out;
}

/*
 * Executes a tensor-scalar elementwise operation.
 */
template <class Op>
bt::Tensor binary_ts(const bt::Tensor& a, float s, Op op) {
  bt::Tensor out(a.shape);
  const int64_t n = a.numel();
  if (n == 0) return out;

  if (a.is_contiguous() && out.is_contiguous()) {
    const float* a_ptr = a.data_ptr();
    float* out_ptr = out.data_ptr();
    for (int64_t i = 0; i < n; ++i) {
      out_ptr[i] = op(a_ptr[i], s);
    }
    return out;
  }

  const int num_dims = a.dim();
  if (num_dims == 0) {
    *out.data_ptr() = op(*a.data_ptr(), s);
    return out;
  }

  recursive_apply_ts(0, num_dims, a.shape, a.data_ptr(), s, out.data_ptr(),
                     a.strides, out.strides, op);
  return out;
}

} /* namespace (anonymous) */

/*
 * Namespace: bt
 * Purpose: Public BareTensor C++ API surface.
 */
namespace bt {

/*
 * Elementwise tensor-tensor addition.
 */
Tensor Tensor::operator+(const Tensor& rhs) const {
  return binary_tt(*this, rhs, ops::Add{});
}

/*
 * Elementwise tensor-scalar addition.
 */
Tensor Tensor::operator+(float rhs) const {
  return binary_ts(*this, rhs, ops::Add{});
}

/*
 * Elementwise tensor-tensor subtraction.
 */
Tensor Tensor::operator-(const Tensor& rhs) const {
  return binary_tt(*this, rhs, ops::Sub{});
}

/*
 * Elementwise tensor-scalar subtraction.
 */
Tensor Tensor::operator-(float rhs) const {
  return binary_ts(*this, rhs, ops::Sub{});
}

/*
 * Elementwise tensor-tensor multiplication.
 */
Tensor Tensor::operator*(const Tensor& rhs) const {
  return binary_tt(*this, rhs, ops::Mul{});
}

/*
 * Elementwise tensor-scalar multiplication.
 */
Tensor Tensor::operator*(float rhs) const {
  return binary_ts(*this, rhs, ops::Mul{});
}

/*
 * Elementwise tensor-tensor division.
 */
Tensor Tensor::operator/(const Tensor& rhs) const {
  return binary_tt(*this, rhs, ops::Div{});
}

/*
 * Elementwise tensor-scalar division.
 */
Tensor Tensor::operator/(float rhs) const {
  return binary_ts(*this, rhs, ops::Div{});
}

} /* namespace bt */
