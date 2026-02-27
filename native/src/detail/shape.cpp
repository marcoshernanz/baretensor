/*
 * File: native/src/detail/shape.cpp
 * Purpose: Implements shape/stride validation and utility helpers.
 */

#include "bt/detail/shape.h"

#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

#include "bt/detail/dims.h"
#include "bt/detail/format.h"

/*
 * Namespace: bt::detail
 * Purpose: Internal reusable implementation helpers.
 */
namespace bt::detail {

/*
 * Computes contiguous strides for the provided shape.
 */
std::vector<int64_t> contiguous_strides(const std::vector<int64_t> &shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (size_t i = shape.size(); i > 1; --i) {
    strides[i - 2] = strides[i - 1] * shape[i - 1];
  }

  return strides;
}

/*
 * Validates shape dimensions and computes total element count.
 */
int64_t checked_numel(const std::vector<int64_t> &shape) {
  int64_t n = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    const int64_t s = shape[i];
    if (s < 0) {
      std::ostringstream oss;
      oss << "Invalid tensor shape " << shape_to_string(shape) << ": dimension "
          << i << " has negative size " << s << ".";
      throw std::invalid_argument(oss.str());
    }
    if (s != 0 && n > std::numeric_limits<int64_t>::max() / s) {
      std::ostringstream oss;
      oss << "Tensor numel overflow for shape " << shape_to_string(shape)
          << ": partial element count " << n << " cannot be multiplied by " << s
          << " within int64 range.";
      throw std::overflow_error(oss.str());
    }
    n *= s;
  }
  return n;
}

/*
 * Resolves a requested reshape target against an input shape.
 * Supports at most one inferred '-1' dimension and validates total elements.
 */
std::vector<int64_t>
infer_reshape_shape(const std::vector<int64_t> &input_shape,
                    const std::vector<int64_t> &requested_shape) {
  const int64_t input_numel = checked_numel(input_shape);

  int64_t known_numel = 1;
  std::optional<size_t> inferred_dim_index;
  for (size_t i = 0; i < requested_shape.size(); ++i) {
    const int64_t d = requested_shape[i];

    if (d == -1) {
      if (inferred_dim_index.has_value()) {
        throw std::invalid_argument("Invalid reshape target " +
                                    shape_to_string(requested_shape) +
                                    ": at most one '-1' dimension is allowed.");
      }
      inferred_dim_index = i;
      continue;
    }

    if (d < -1) {
      throw std::invalid_argument(
          "Invalid reshape target " + shape_to_string(requested_shape) +
          ": dimension " + std::to_string(i) + " has invalid size " +
          std::to_string(d) + ".");
    }

    if (d != 0 && known_numel > std::numeric_limits<int64_t>::max() / d) {
      throw std::overflow_error("Reshape target numel overflow for shape " +
                                shape_to_string(requested_shape) + ".");
    }
    known_numel *= d;
  }

  if (!inferred_dim_index.has_value()) {
    if (known_numel != input_numel) {
      throw std::invalid_argument(
          "Invalid reshape from " + shape_to_string(input_shape) + " to " +
          shape_to_string(requested_shape) + ": element counts differ (" +
          std::to_string(input_numel) + " vs " + std::to_string(known_numel) +
          ").");
    }
    return requested_shape;
  }

  if (known_numel == 0) {
    throw std::invalid_argument(
        "Invalid reshape target " + shape_to_string(requested_shape) +
        ": cannot infer '-1' when known dimensions multiply to zero.");
  }

  if (input_numel % known_numel != 0) {
    throw std::invalid_argument(
        "Invalid reshape from " + shape_to_string(input_shape) + " to " +
        shape_to_string(requested_shape) + ": cannot infer '-1' because " +
        std::to_string(input_numel) + " is not divisible by " +
        std::to_string(known_numel) + ".");
  }

  std::vector<int64_t> resolved_shape(requested_shape);
  resolved_shape[*inferred_dim_index] = input_numel / known_numel;
  return resolved_shape;
}

/*
 * Computes view strides for a target shape if the current layout is viewable
 * without copying; returns std::nullopt when layout compatibility is not met.
 */
std::optional<std::vector<int64_t>>
infer_view_strides(const std::vector<int64_t> &input_shape,
                   const std::vector<int64_t> &input_strides,
                   const std::vector<int64_t> &target_shape) {
  if (input_shape.size() != input_strides.size()) {
    return std::nullopt;
  }

  const int64_t input_numel = checked_numel(input_shape);
  const int64_t target_numel = checked_numel(target_shape);
  if (input_numel != target_numel) {
    return std::nullopt;
  }

  if (target_shape.empty()) {
    return std::vector<int64_t>{};
  }

  if (input_numel == 0) {
    return contiguous_strides(target_shape);
  }

  if (input_shape.empty()) {
    return contiguous_strides(target_shape);
  }

  std::vector<int64_t> target_strides(target_shape.size(), 0);
  int64_t target_dim = static_cast<int64_t>(target_shape.size()) - 1;
  int64_t chunk_base_stride = input_strides.back();
  int64_t input_chunk_numel = 1;
  int64_t target_chunk_numel = 1;

  for (int64_t input_dim = static_cast<int64_t>(input_shape.size()) - 1;
       input_dim >= 0; --input_dim) {
    input_chunk_numel *= input_shape[dim_to_index(input_dim)];

    const bool is_chunk_boundary =
        (input_dim == 0) ||
        ((input_shape[dim_to_index(input_dim - 1)] != 1) &&
         (input_strides[dim_to_index(input_dim - 1)] !=
          input_chunk_numel * chunk_base_stride));

    if (!is_chunk_boundary) {
      continue;
    }

    while (target_dim >= 0 &&
           (target_chunk_numel < input_chunk_numel ||
            target_shape[dim_to_index(target_dim)] == 1)) {
      target_strides[dim_to_index(target_dim)] =
          target_chunk_numel * chunk_base_stride;
      target_chunk_numel *= target_shape[dim_to_index(target_dim)];
      --target_dim;
    }

    if (target_chunk_numel != input_chunk_numel) {
      return std::nullopt;
    }

    if (input_dim > 0) {
      chunk_base_stride = input_strides[dim_to_index(input_dim - 1)];
      input_chunk_numel = 1;
      target_chunk_numel = 1;
    }
  }

  if (target_dim != -1) {
    return std::nullopt;
  }

  return target_strides;
}

} /* namespace bt::detail */
