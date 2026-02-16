#include <stdexcept>
#include <vector>

#include "bt/tensor.h"

template <class Op>
void recursive_apply_tt(int dim, int last_dim,
                        const std::vector<int64_t>& shape,
                        const std::vector<float>& a,
                        const std::vector<float>& b, std::vector<float>& out,
                        int64_t offset_a, int64_t offset_b, int64_t offset_out,
                        const std::vector<int64_t>& stride_a,
                        const std::vector<int64_t>& stride_b,
                        const std::vector<int64_t>& stride_out, Op op) {
  if (shape.size() == 0) {
    out[offset_out] = op(a[offset_a], b[offset_b]);
    return;
  }

  if (dim == last_dim) {
    for (int64_t i = 0; i < shape[dim]; i++) {
      out[offset_out] = op(a[offset_a], b[offset_b]);
      offset_a += stride_a[dim];
      offset_b += stride_b[dim];
      offset_out += stride_out[dim];
    }
    return;
  }

  for (int64_t i = 0; i < shape[dim]; i++) {
    recursive_apply(dim + 1, last_dim, shape, a, b, out, offset_a, offset_b,
                    offset_out, stride_a, stride_b, stride_out, op);
    offset_a += stride_a[dim];
    offset_b += stride_b[dim];
    offset_out += stride_out[dim];
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

  recursive_apply_tt(0, a.dim(), a.shape, a, b, out, a.storage_offset,
                     b.storage_offset, out.storage_offset, a.strides, b.strides,
                     out.strides, op);
  return out;
}