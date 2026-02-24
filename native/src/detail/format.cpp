/*
 * File: native/src/detail/format.cpp
 * Purpose: Implements shared formatting helpers for internal diagnostics.
 */

#include "bt/detail/format.h"

#include <sstream>

/*
 * Namespace: bt::detail
 * Purpose: Internal reusable implementation helpers.
 */
namespace bt::detail {

/*
 * Converts a shape vector into a human-readable string.
 */
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

} /* namespace bt::detail */
