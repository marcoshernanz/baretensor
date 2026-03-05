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
 * Normalizes and validates a single dimension for an operation.
 */
[[nodiscard]] int64_t normalize_dim_checked(std::string_view operation_name,
                                            const std::vector<int64_t> &shape,
                                            int64_t dim, std::string_view dim_name);

/*
 * Normalizes and validates a full permutation for an operation.
 */
[[nodiscard]] std::vector<int64_t>
normalize_permutation_checked(std::string_view operation_name,
                              const std::vector<int64_t> &shape,
                              const std::vector<int64_t> &dims);

/*
 * Builds an identity axis order [0, 1, ..., rank - 1].
 */
[[nodiscard]] std::vector<int64_t> make_axis_order(size_t rank);

/*
 * Inverts a permutation so inverse[dims[i]] = i.
 */
[[nodiscard]] std::vector<int64_t> invert_permutation(const std::vector<int64_t> &dims);

} /* namespace bt::detail */
