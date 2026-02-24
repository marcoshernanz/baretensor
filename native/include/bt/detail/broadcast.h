#pragma once

#include <cstdint>
#include <vector>

namespace bt::detail {

[[nodiscard]] std::vector<int64_t> infer_broadcast_shape(
    const std::vector<int64_t>& a_shape, const std::vector<int64_t>& b_shape);

[[nodiscard]] std::vector<int64_t> aligned_broadcast_strides(
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& in_strides,
    const std::vector<int64_t>& out_shape);

}  // namespace bt::detail
