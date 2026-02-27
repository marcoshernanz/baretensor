/*
 * File: native/src/detail/dims.cpp
 * Purpose: Implements shared dimension normalization and permutation helpers.
 */

#include "bt/detail/dims.h"

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
 * Normalizes a possibly negative dimension using Python-style indexing.
 */
[[nodiscard]] int64_t normalize_dim(const int64_t dim,
                                    const int64_t rank) noexcept {
  return dim < 0 ? dim + rank : dim;
}

} // namespace

/*
 * Normalizes and validates a single dimension for an operation.
 */
int64_t normalize_dim_checked(std::string_view operation_name,
                              const std::vector<int64_t>& shape,
                              const int64_t dim,
                              std::string_view dim_name) {
  const int64_t rank = static_cast<int64_t>(shape.size());
  const int64_t normalized = normalize_dim(dim, rank);
  if (normalized >= 0 && normalized < rank) {
    return normalized;
  }

  std::ostringstream oss;
  oss << operation_name << " failed for tensor with shape "
      << shape_to_string(shape) << ": " << dim_name << "=" << dim
      << " is out of range for rank " << rank << ".";
  throw std::invalid_argument(oss.str());
}

/*
 * Normalizes and validates a full permutation for an operation.
 */
std::vector<int64_t> normalize_permutation_checked(
    std::string_view operation_name, const std::vector<int64_t>& shape,
    const std::vector<int64_t>& dims) {
  const int64_t rank = static_cast<int64_t>(shape.size());
  if (static_cast<int64_t>(dims.size()) != rank) {
    std::ostringstream oss;
    oss << operation_name << " failed for tensor with shape "
        << shape_to_string(shape) << ": expected " << rank
        << " dims but got " << dims.size() << ".";
    throw std::invalid_argument(oss.str());
  }

  std::vector<int64_t> normalized_dims(dims.size(), 0);
  std::vector<bool> seen(dims.size(), false);
  for (size_t i = 0; i < dims.size(); ++i) {
    const int64_t dim = dims[i];
    const int64_t normalized = normalize_dim(dim, rank);
    if (normalized < 0 || normalized >= rank) {
      std::ostringstream oss;
      oss << operation_name << " failed for tensor with shape "
          << shape_to_string(shape) << ": dims[" << i << "]=" << dim
          << " is out of range for rank " << rank << ".";
      throw std::invalid_argument(oss.str());
    }
    if (seen[static_cast<size_t>(normalized)]) {
      std::ostringstream oss;
      oss << operation_name << " failed for tensor with shape "
          << shape_to_string(shape) << ": dimension " << normalized
          << " appears more than once.";
      throw std::invalid_argument(oss.str());
    }
    normalized_dims[i] = normalized;
    seen[static_cast<size_t>(normalized)] = true;
  }

  return normalized_dims;
}

} /* namespace bt::detail */
