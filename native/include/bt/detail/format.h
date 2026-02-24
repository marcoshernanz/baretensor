/*
 * File: native/include/bt/detail/format.h
 * Purpose: Declares shared formatting helpers for internal diagnostics.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

/*
 * Namespace: bt::detail
 * Purpose: Internal reusable implementation helpers.
 */
namespace bt::detail {

/*
 * Converts a shape vector into a human-readable string.
 */
[[nodiscard]] std::string shape_to_string(const std::vector<int64_t>& shape);

} /* namespace bt::detail */
