#include "bt/ops.h"

#include <stdexcept>
#include <vector>

#include "bt/tensor.h"

template <class Op>
void recursive_apply_tt(int dim, int num_dims,
                        const std::vector<int64_t>& shape, const float* a,
                        const float* b, float* out,
                        const std::vector<int64_t>& stride_a,
                        const std::vector<int64_t>& stride_b,
                        const std::vector<int64_t>& stride_out, Op op) {
  if (shape.size() == 0) {
    *out = op(*a, *b);
    return;
  }

  if (dim == num_dims - 1) {
    for (int64_t i = 0; i < shape[dim]; i++) {
      *out = op(*a, *b);
      a += stride_a[dim];
      b += stride_b[dim];
      out += stride_out[dim];
    }
    return;
  }

  for (int64_t i = 0; i < shape[dim]; i++) {
    recursive_apply_tt(dim + 1, num_dims, shape, a, b, out, stride_a, stride_b,
                       stride_out, op);
    a += stride_a[dim];
    b += stride_b[dim];
    out += stride_out[dim];
  }
}

template <class Op>
void recursive_apply_ts(int dim, int num_dims,
                        const std::vector<int64_t>& shape, const float* a,
                        float s, float* out,
                        const std::vector<int64_t>& stride_a,
                        const std::vector<int64_t>& stride_out, Op op) {
  if (shape.size() == 0) {
    *out = op(*a, s);
    return;
  }

  if (dim == num_dims - 1) {
    for (int64_t i = 0; i < shape[dim]; i++) {
      *out = op(*a, s);
      a += stride_a[dim];
      out += stride_out[dim];
    }
    return;
  }

  for (int64_t i = 0; i < shape[dim]; i++) {
    recursive_apply_ts(dim + 1, num_dims, shape, a, s, out, stride_a,
                       stride_out, op);
    a += stride_a[dim];
    out += stride_out[dim];
  }
}

template <class Op>
bt::Tensor binary_tt(const bt::Tensor& a, const bt::Tensor& b, Op op) {
  if (a.shape != b.shape) throw std::runtime_error("shape mismatch");

  bt::Tensor out(a.shape);
  int64_t n = a.numel();
  if (n == 0) return out;

  if (a.is_contiguous() && b.is_contiguous() && out.is_contiguous()) {
    const float* a_ptr = a.data_ptr();
    const float* b_ptr = b.data_ptr();
    float* out_ptr = out.data_ptr();
    for (int64_t i = 0; i < n; i++) {
      out_ptr[i] = op(a_ptr[i], b_ptr[i]);
    }
    return out;
  }

  recursive_apply_tt(0, a.dim(), a.shape, a.data_ptr(), b.data_ptr(),
                     out.data_ptr(), a.stride, b.stride, out.stride, op);
  return out;
}

template <class Op>
bt::Tensor binary_ts(const bt::Tensor& a, float s, Op op) {
  bt::Tensor out(a.shape);
  int64_t n = a.numel();
  if (n == 0) return out;

  if (a.is_contiguous() && out.is_contiguous()) {
    const float* a_ptr = a.data_ptr();
    float* out_ptr = out.data_ptr();
    for (int64_t i = 0; i < n; i++) {
      out_ptr[i] = op(a_ptr[i], s);
    }
    return out;
  }

  recursive_apply_ts(0, a.dim(), a.shape, a.data_ptr(), s, out.data_ptr(),
                     a.stride, out.stride, op);
  return out;
}

bt::Tensor bt::Tensor::operator+(const bt::Tensor& rhs) const {
  return binary_tt(*this, rhs, bt::ops::Add{});
}

bt::Tensor bt::Tensor::operator+(float s) const {
  return binary_ts(*this, s, bt::ops::Add{});
}

bt::Tensor bt::Tensor::operator-(const bt::Tensor& rhs) const {
  return binary_tt(*this, rhs, bt::ops::Sub{});
}

bt::Tensor bt::Tensor::operator-(float s) const {
  return binary_ts(*this, s, bt::ops::Sub{});
}

bt::Tensor bt::Tensor::operator*(const bt::Tensor& rhs) const {
  return binary_tt(*this, rhs, bt::ops::Mul{});
}

bt::Tensor bt::Tensor::operator*(float s) const {
  return binary_ts(*this, s, bt::ops::Mul{});
}

bt::Tensor bt::Tensor::operator/(const bt::Tensor& rhs) const {
  return binary_tt(*this, rhs, bt::ops::Div{});
}

bt::Tensor bt::Tensor::operator/(float s) const {
  return binary_ts(*this, s, bt::ops::Div{});
}