/*
 * File: native/src/detail/tensor_copy.cpp
 * Purpose: Implements shared helpers for copying tensor values across layouts.
 */

#include "bt/detail/tensor_copy.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

/*
 * Recursively copies data from a strided source layout into a strided
 * destination layout over a shared logical shape.
 */
void recursive_copy(const size_t dim, const size_t ndim, const std::vector<int64_t> &shape,
                    const std::byte *src, std::byte *dst, const size_t itemsize,
                    const std::vector<int64_t> &src_strides,
                    const std::vector<int64_t> &dst_strides) {
  if (shape[dim] == 0) {
    return;
  }

  if (dim == ndim - 1) {
    for (int64_t i = 0; i < shape[dim]; ++i) {
      std::memcpy(dst, src, itemsize);
      src += src_strides[dim] * static_cast<int64_t>(itemsize);
      dst += dst_strides[dim] * static_cast<int64_t>(itemsize);
    }
    return;
  }

  for (int64_t i = 0; i < shape[dim]; ++i) {
    recursive_copy(dim + 1, ndim, shape, src, dst, itemsize, src_strides, dst_strides);
    src += src_strides[dim] * static_cast<int64_t>(itemsize);
    dst += dst_strides[dim] * static_cast<int64_t>(itemsize);
  }
}

} // namespace

namespace bt::detail {

/*
 * Copies tensor values from src into dst assuming matching shapes and dtype.
 */
void copy_tensor_values(const bt::Tensor &src, bt::Tensor &dst) {
  if (src.shape != dst.shape) {
    throw std::runtime_error("copy_tensor_values failed: source and "
                             "destination tensor shapes do not match.");
  }
  if (src.dtype() != dst.dtype()) {
    throw std::runtime_error("copy_tensor_values failed: source and "
                             "destination tensor dtypes do not match.");
  }

  const size_t itemsize = bt::scalar_type_itemsize(src.dtype());
  if (src.ndim() == 0) {
    std::memcpy(dst.raw_data_ptr(), src.raw_data_ptr(), itemsize);
    return;
  }

  recursive_copy(0, src.shape.size(), src.shape, src.raw_data_ptr(), dst.raw_data_ptr(), itemsize,
                 src.strides, dst.strides);
}

} // namespace bt::detail
