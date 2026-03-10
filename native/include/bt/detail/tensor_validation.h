/*
 * File: native/include/bt/detail/tensor_validation.h
 * Purpose: Shared tensor metadata validation helpers for tensor kernels.
 */

#pragma once

#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "bt/detail/format.h"
#include "bt/tensor.h"

namespace bt::detail {

/*
 * Validates tensor metadata invariants required by low-level tensor routines.
 */
inline void validate_copy_metadata(const bt::Tensor &tensor, const std::string &operation_name) {
  if (!tensor.storage) {
    throw std::invalid_argument(operation_name + " failed: tensor storage is null.");
  }

  if (tensor.storage_offset < 0) {
    throw std::invalid_argument(operation_name + " failed for tensor with shape " +
                                bt::detail::shape_to_string(tensor.shape) +
                                ": storage offset must be non-negative, got " +
                                std::to_string(tensor.storage_offset) + ".");
  }

  if (tensor.shape.size() != tensor.strides.size()) {
    throw std::invalid_argument(operation_name + " failed for tensor with shape " +
                                bt::detail::shape_to_string(tensor.shape) + ": shape rank " +
                                std::to_string(tensor.shape.size()) +
                                " does not match stride rank " +
                                std::to_string(tensor.strides.size()) + ".");
  }
}

/*
 * Throws unless tensor has the expected dtype.
 */
inline void ensure_dtype(const bt::Tensor &tensor, const bt::ScalarType expected,
                         const std::string_view operation_name,
                         const std::string_view argument_name = "tensor") {
  if (tensor.dtype() == expected) {
    return;
  }

  std::ostringstream oss;
  oss << operation_name << " failed for tensor with shape "
      << bt::detail::shape_to_string(tensor.shape) << ": " << argument_name << " must have dtype "
      << bt::scalar_type_name(expected) << ", got " << bt::scalar_type_name(tensor.dtype()) << ".";
  throw std::invalid_argument(oss.str());
}

/*
 * Throws unless tensor has dtype float32.
 */
inline void ensure_float32(const bt::Tensor &tensor, const std::string_view operation_name,
                           const std::string_view argument_name = "tensor") {
  ensure_dtype(tensor, bt::ScalarType::kFloat32, operation_name, argument_name);
}

/*
 * Throws when two tensors have different dtypes.
 */
inline void ensure_same_dtype(const bt::Tensor &lhs, const bt::Tensor &rhs,
                              const std::string_view operation_name) {
  if (lhs.dtype() == rhs.dtype()) {
    return;
  }

  std::ostringstream oss;
  oss << operation_name << " failed for tensors with shapes "
      << bt::detail::shape_to_string(lhs.shape) << " and " << bt::detail::shape_to_string(rhs.shape)
      << ": operands must have the same dtype, but got " << bt::scalar_type_name(lhs.dtype())
      << " and " << bt::scalar_type_name(rhs.dtype()) << ".";
  throw std::invalid_argument(oss.str());
}

/*
 * Throws unless tensor has a floating-point dtype.
 */
inline void ensure_floating_dtype(const bt::Tensor &tensor, const std::string_view operation_name,
                                  const std::string_view argument_name = "tensor") {
  if (bt::is_floating_point(tensor.dtype())) {
    return;
  }

  std::ostringstream oss;
  oss << operation_name << " failed for tensor with shape "
      << bt::detail::shape_to_string(tensor.shape) << ": " << argument_name
      << " must have a floating-point dtype, but got " << bt::scalar_type_name(tensor.dtype())
      << ".";
  throw std::invalid_argument(oss.str());
}

} // namespace bt::detail
