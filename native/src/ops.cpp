#include "bt/ops.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "bt/tensor.h"

namespace {

[[nodiscard]] std::vector<int64_t> infer_broadcast_shape(
    const std::vector<int64_t>& a_shape, const std::vector<int64_t>& b_shape) {
  const size_t out_rank = std::max(a_shape.size(), b_shape.size());
  std::vector<int64_t> out(out_rank, 1);

  for (size_t i = 0; i < out_rank; ++i) {
    const size_t out_i = out_rank - 1 - i;

    const int64_t a_dim =
        (i < a_shape.size()) ? a_shape[a_shape.size() - 1 - i] : int64_t{1};
    const int64_t b_dim =
        (i < b_shape.size()) ? b_shape[b_shape.size() - 1 - i] : int64_t{1};

    if (a_dim == b_dim) {
      out[out_i] = a_dim;
    } else if (a_dim == 1) {
      out[out_i] = b_dim;
    } else if (b_dim == 1) {
      out[out_i] = a_dim;
    } else {
      throw std::invalid_argument("shape mismatch");
    }
  }

  return out;
}

[[nodiscard]] std::vector<int64_t> aligned_broadcast_strides(
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& in_strides,
    const std::vector<int64_t>& out_shape) {
  const size_t out_rank = out_shape.size();
  const size_t in_rank = in_shape.size();
  std::vector<int64_t> out_strides(out_rank, 0);

  for (size_t i = 0; i < out_rank; ++i) {
    const size_t out_i = out_rank - 1 - i;
    if (i >= in_rank) {
      out_strides[out_i] = 0;
      continue;
    }

    const size_t in_i = in_rank - 1 - i;
    const int64_t in_dim = in_shape[in_i];
    const int64_t out_dim = out_shape[out_i];

    if (in_dim == out_dim) {
      out_strides[out_i] = in_strides[in_i];
    } else if (in_dim == 1) {
      out_strides[out_i] = 0;
    } else {
      throw std::invalid_argument("shape mismatch");
    }
  }

  return out_strides;
}

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

template <class Op>
bt::Tensor binary_tt(const bt::Tensor& a, const bt::Tensor& b, Op op) {
  const std::vector<int64_t> out_shape =
      infer_broadcast_shape(a.shape, b.shape);
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
      aligned_broadcast_strides(a.shape, a.strides, out_shape);
  const std::vector<int64_t> stride_b =
      aligned_broadcast_strides(b.shape, b.strides, out_shape);

  recursive_apply_tt(0, num_dims, out_shape, a.data_ptr(), b.data_ptr(),
                     out.data_ptr(), stride_a, stride_b, out.strides, op);
  return out;
}

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

}  // namespace

namespace bt {

Tensor Tensor::operator+(const Tensor& rhs) const {
  return binary_tt(*this, rhs, ops::Add{});
}

Tensor Tensor::operator+(float rhs) const {
  return binary_ts(*this, rhs, ops::Add{});
}

Tensor Tensor::operator-(const Tensor& rhs) const {
  return binary_tt(*this, rhs, ops::Sub{});
}

Tensor Tensor::operator-(float rhs) const {
  return binary_ts(*this, rhs, ops::Sub{});
}

Tensor Tensor::operator*(const Tensor& rhs) const {
  return binary_tt(*this, rhs, ops::Mul{});
}

Tensor Tensor::operator*(float rhs) const {
  return binary_ts(*this, rhs, ops::Mul{});
}

Tensor Tensor::operator/(const Tensor& rhs) const {
  return binary_tt(*this, rhs, ops::Div{});
}

Tensor Tensor::operator/(float rhs) const {
  return binary_ts(*this, rhs, ops::Div{});
}

}  // namespace bt
