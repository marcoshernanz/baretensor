#include "bt/detail/format.h"

#include <sstream>

namespace bt::detail {

std::string shape_to_string(const std::vector<int64_t>& shape) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) oss << ", ";
    oss << shape[i];
  }
  oss << "]";
  return oss.str();
}

}  // namespace bt::detail
