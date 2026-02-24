/*
 * File: native/include/bt/detail/broadcast.h
 * Purpose: Declares internal broadcasting shape and stride alignment helpers.
 */

#pragma once

#include <cstdint>
#include <vector>

/*
 * Namespace: bt::detail
 * Purpose: Internal reusable implementation helpers.
 */
namespace bt::detail {

/*
 * Infers the broadcasted output shape for two input shapes.
 */
[[nodiscard]] std::vector<int64_t> infer_broadcast_shape(
    const std::vector<int64_t>& a_shape, const std::vector<int64_t>& b_shape);

/*
 * Aligns input strides to an output broadcast shape using zero-stride expansion.
 */
[[nodiscard]] std::vector<int64_t> aligned_broadcast_strides(
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& in_strides,
    const std::vector<int64_t>& out_shape);

} /* namespace bt::detail */
