/*
 * File: native/src/detail/tensor_copy.cpp
 * Purpose: Implements shared helpers for copying tensor values across layouts.
 */

#include "bt/detail/tensor_copy.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

/*
 * Recursively copies data from a strided source layout into a strided
 * destination layout over a shared logical shape.
 */
void recursive_copy(const size_t dim, const size_t ndim,
                    const std::vector<int64_t> &shape, const float *src,
                    float *dst, const std::vector<int64_t> &src_strides,
                    const std::vector<int64_t> &dst_strides) {
  if (shape[dim] == 0) {
    return;
  }

  if (dim == ndim - 1) {
    for (int64_t i = 0; i < shape[dim]; ++i) {
      *dst = *src;
      src += src_strides[dim];
      dst += dst_strides[dim];
    }
    return;
  }

  for (int64_t i = 0; i < shape[dim]; ++i) {
    recursive_copy(dim + 1, ndim, shape, src, dst, src_strides, dst_strides);
    src += src_strides[dim];
    dst += dst_strides[dim];
  }
}

} // namespace

namespace bt::detail {

/*
 * Copies tensor values from src into dst assuming matching shapes.
 */
void copy_tensor_values(const bt::Tensor &src, bt::Tensor &dst) {
  if (src.shape != dst.shape) {
    throw std::runtime_error("copy_tensor_values failed: source and "
                             "destination tensor shapes do not match.");
  }

  if (src.ndim() == 0) {
    *dst.data_ptr() = *src.data_ptr();
    return;
  }

  recursive_copy(0, src.shape.size(), src.shape, src.data_ptr(), dst.data_ptr(),
                 src.strides, dst.strides);
}

} // namespace bt::detail
