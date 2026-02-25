/*
 * File: native/include/bt/detail/shape.h
 * Purpose: Declares internal shape/stride utilities and validation helpers.
 */

#pragma once

#include <cstdint>
#include <vector>

#include "bt/tensor.h"

/*
 * Namespace: bt::detail
 * Purpose: Internal reusable implementation helpers.
 */
namespace bt::detail {

/*
 * Computes contiguous strides for the provided shape.
 */
[[nodiscard]] std::vector<int64_t> contiguous_strides(
    const std::vector<int64_t>& shape);

/*
 * Validates shape dimensions and computes total element count.
 */
[[nodiscard]] int64_t checked_numel(const std::vector<int64_t>& shape);

/*
 * TODO
 */
[[nodiscard]] std::vector<int64_t> check_shape(bt::Tensor& tensor,
                                               std::vector<int64_t>& shape);

} /* namespace bt::detail */
