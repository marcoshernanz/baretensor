/*
 * File: native/include/bt/detail/dims.h
 * Purpose: Declares shared dimension normalization and permutation utilities.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

/*
 * Namespace: bt::detail
 * Purpose: Internal reusable implementation helpers.
 */
namespace bt::detail {

/*
 * Converts a signed dimension value to a vector index.
 */
[[nodiscard]] size_t dim_to_index(int64_t dim) noexcept;

/*
 * Builds an identity permutation [0, 1, ..., rank - 1].
 */
[[nodiscard]] std::vector<int64_t> identity_permutation(size_t rank);

/*
 * Normalizes and validates a single dimension for an operation.
 */
[[nodiscard]] int64_t normalize_dim_checked(
    std::string_view operation_name, const std::vector<int64_t>& shape,
    int64_t dim, std::string_view dim_name);

/*
 * Normalizes and validates a full permutation for an operation.
 */
[[nodiscard]] std::vector<int64_t> normalize_permutation_checked(
    std::string_view operation_name, const std::vector<int64_t>& shape,
    const std::vector<int64_t>& dims);

} /* namespace bt::detail */
