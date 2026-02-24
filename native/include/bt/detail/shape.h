#pragma once

#include <cstdint>
#include <vector>

namespace bt::detail {

[[nodiscard]] std::vector<int64_t> contiguous_strides(
    const std::vector<int64_t>& shape);

[[nodiscard]] int64_t checked_numel(const std::vector<int64_t>& shape);

}  // namespace bt::detail
