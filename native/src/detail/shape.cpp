#include "bt/detail/shape.h"

#include <limits>
#include <sstream>
#include <stdexcept>

#include "bt/detail/format.h"

namespace bt::detail {

std::vector<int64_t> contiguous_strides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (size_t i = shape.size(); i > 1; --i) {
    strides[i - 2] = strides[i - 1] * shape[i - 1];
  }

  return strides;
}

int64_t checked_numel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    const int64_t s = shape[i];
    if (s < 0) {
      std::ostringstream oss;
      oss << "Invalid tensor shape " << shape_to_string(shape) << ": dimension "
          << i << " has negative size " << s << ".";
      throw std::invalid_argument(oss.str());
    }
    if (s != 0 && n > std::numeric_limits<int64_t>::max() / s) {
      std::ostringstream oss;
      oss << "Tensor numel overflow for shape " << shape_to_string(shape)
          << ": partial element count " << n << " cannot be multiplied by "
          << s << " within int64 range.";
      throw std::overflow_error(oss.str());
    }
    n *= s;
  }
  return n;
}

}  // namespace bt::detail
