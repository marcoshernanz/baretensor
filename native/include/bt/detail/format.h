#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace bt::detail {

[[nodiscard]] std::string shape_to_string(const std::vector<int64_t>& shape);

}  // namespace bt::detail
