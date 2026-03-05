/*
 * File: native/include/bt/detail/tensor_validation.h
 * Purpose: Shared tensor metadata validation helpers for tensor kernels.
 */

#pragma once

#include <stdexcept>
#include <string>

#include "bt/detail/format.h"
#include "bt/tensor.h"

namespace bt::detail {

/*
 * Validates tensor metadata invariants required by low-level copy routines.
 */
inline void validate_copy_metadata(const bt::Tensor &tensor,
                                   const std::string &operation_name) {
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
                                bt::detail::shape_to_string(tensor.shape) +
                                ": shape rank " + std::to_string(tensor.shape.size()) +
                                " does not match stride rank " +
                                std::to_string(tensor.strides.size()) + ".");
  }
}

} // namespace bt::detail
