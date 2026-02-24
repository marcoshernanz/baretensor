/*
 * File: native/src/detail/broadcast.cpp
 * Purpose: Implements shape inference and stride alignment for broadcasting.
 */

#include "bt/detail/broadcast.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

#include "bt/detail/format.h"

/*
 * Namespace: bt::detail
 * Purpose: Internal reusable implementation helpers.
 */
namespace bt::detail {

/*
 * Namespace: (anonymous)
 * Purpose: Private implementation details local to this translation unit.
 */
namespace {

/*
 * Throws a detailed broadcast mismatch error for incompatible dimensions.
 */
[[noreturn]] void throw_broadcast_mismatch(
    const std::vector<int64_t>& a_shape, const std::vector<int64_t>& b_shape,
    size_t out_i, int64_t a_dim, int64_t b_dim) {
  const size_t out_rank = std::max(a_shape.size(), b_shape.size());
  const size_t axis_from_right = out_rank - out_i;

  std::ostringstream oss;
  oss << "Cannot broadcast shapes " << shape_to_string(a_shape) << " and "
      << shape_to_string(b_shape)
      << ": incompatible dimension at axis -" << axis_from_right
      << " (from right), got " << a_dim << " and " << b_dim << ".";
  throw std::invalid_argument(oss.str());
}

} /* namespace (anonymous) */

/*
 * Infers the broadcasted output shape for two input shapes.
 */
std::vector<int64_t> infer_broadcast_shape(
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
      throw_broadcast_mismatch(a_shape, b_shape, out_i, a_dim, b_dim);
    }
  }

  return out;
}

/*
 * Aligns input strides to an output broadcast shape.
 */
std::vector<int64_t> aligned_broadcast_strides(
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& in_strides,
    const std::vector<int64_t>& out_shape) {
  if (in_shape.size() != in_strides.size()) {
    std::ostringstream oss;
    oss << "Tensor metadata invariant violation: shape rank " << in_shape.size()
        << " does not match stride rank " << in_strides.size() << ".";
    throw std::invalid_argument(oss.str());
  }

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
      std::ostringstream oss;
      oss << "Internal broadcast stride alignment error: cannot align input "
          << "shape " << shape_to_string(in_shape) << " with output shape "
          << shape_to_string(out_shape) << " at axis -" << (out_rank - out_i)
          << " (from right), got " << in_dim << " and " << out_dim << ".";
      throw std::invalid_argument(oss.str());
    }
  }

  return out_strides;
}

} /* namespace bt::detail */
