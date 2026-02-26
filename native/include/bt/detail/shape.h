/*
 * File: native/include/bt/detail/shape.h
 * Purpose: Declares internal shape, stride, and reshape validation helpers.
 */

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

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
 * Resolves a requested reshape target against an input shape.
 * Supports at most one inferred '-1' dimension and validates total elements.
 */
[[nodiscard]] std::vector<int64_t> infer_reshape_shape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& requested_shape);

/*
 * Computes view strides for a target shape if the current layout is viewable
 * without copying; returns std::nullopt when layout compatibility is not met.
 */
[[nodiscard]] std::optional<std::vector<int64_t>> infer_view_strides(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    const std::vector<int64_t>& target_shape);

} /* namespace bt::detail */
