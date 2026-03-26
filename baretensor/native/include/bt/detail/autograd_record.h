/*
 * File: native/include/bt/detail/autograd_record.h
 * Purpose: Shared helpers for deciding when to attach autograd nodes.
 */

#pragma once

#include <sstream>
#include <stdexcept>
#include <string_view>

#include "bt/tensor.h"

namespace bt::detail {

/*
 * Returns whether a unary operation should record an autograd node.
 */
[[nodiscard]] inline bool should_record_unary(const bt::Tensor &input) {
  return bt::autograd::is_grad_enabled() && input.requires_grad();
}

/*
 * Returns whether a binary operation should record an autograd node.
 */
[[nodiscard]] inline bool should_record_binary(const bt::Tensor &lhs,
                                               const bt::Tensor &rhs) {
  return bt::autograd::is_grad_enabled() && (lhs.requires_grad() || rhs.requires_grad());
}

/*
 * Throws a consistent fail-loud error for unsupported autograd ops.
 */
[[noreturn]] inline void throw_autograd_not_implemented(const std::string_view op_name) {
  std::ostringstream oss;
  oss << "Autograd support for " << op_name << " is not implemented yet.";
  throw std::runtime_error(oss.str());
}

} // namespace bt::detail
