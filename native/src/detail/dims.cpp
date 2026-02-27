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
 * Converts a signed dimension value to a vector index.
 */
size_t dim_to_index(const int64_t dim) noexcept {
  return static_cast<size_t>(dim);
}

/*
 * Builds an identity permutation [0, 1, ..., rank - 1].
 */
std::vector<int64_t> identity_permutation(const size_t rank) {
  std::vector<int64_t> dims(rank, 0);
  for (size_t i = 0; i < rank; ++i) {
    dims[i] = static_cast<int64_t>(i);
  }
  return dims;
}

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
    if (seen[dim_to_index(normalized)]) {
      std::ostringstream oss;
      oss << operation_name << " failed for tensor with shape "
          << shape_to_string(shape) << ": dimension " << normalized
          << " appears more than once.";
      throw std::invalid_argument(oss.str());
    }
    normalized_dims[i] = normalized;
    seen[dim_to_index(normalized)] = true;
  }

  return normalized_dims;
}

} /* namespace bt::detail */
