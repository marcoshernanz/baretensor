/*
 * File: native/src/detail/shape.cpp
 * Purpose: Implements shape/stride validation and utility helpers.
 */

#include "bt/detail/shape.h"

#include <limits>
#include <sstream>
#include <stdexcept>

#include "bt/detail/format.h"

/*
 * Namespace: bt::detail
 * Purpose: Internal reusable implementation helpers.
 */
namespace bt::detail {

/*
 * Computes contiguous strides for the provided shape.
 */
std::vector<int64_t> contiguous_strides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (size_t i = shape.size(); i > 1; --i) {
    strides[i - 2] = strides[i - 1] * shape[i - 1];
  }

  return strides;
}

/*
 * Validates shape dimensions and computes total element count.
 */
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
          << ": partial element count " << n << " cannot be multiplied by " << s
          << " within int64 range.";
      throw std::overflow_error(oss.str());
    }
    n *= s;
  }
  return n;
}

/*
 * TODO
 */
[[nodiscard]] std::vector<int64_t> check_shape(bt::Tensor& tensor,
                                               std::vector<int64_t>& shape) {
  int64_t numel = 1;
  int infer_pos = -1;
  for (int i = 0; i < shape.size(); ++i) {
    if (shape[i] == -1 && infer_pos == -1) {
      infer_pos = i;
    } else if (shape[i] == -1) {
      throw std::invalid_argument("invalid shape");
    } else {
      numel *= shape[i];
    }
  }

  if (infer_pos == -1 && numel == tensor.numel()) {
    std::vector<int64_t> new_shape(shape);
    return new_shape;
  } else if (infer_pos == -1) {
    throw std::invalid_argument("invalid shape");
  } else if (tensor.numel() > numel && tensor.numel() % numel == 0) {
    std::vector<int64_t> new_shape(shape);
    new_shape[infer_pos] = tensor.numel() / numel;
    return new_shape;
  } else {
    throw std::invalid_argument("invalid shape");
  }
}

} /* namespace bt::detail */
