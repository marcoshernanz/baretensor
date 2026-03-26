/*
 * File: native/include/bt/detail/tensor_copy.h
 * Purpose: Declares shared helpers for copying tensor values across layouts.
 */

#pragma once

#include "bt/tensor.h"

namespace bt::detail {

/*
 * Copies logical tensor values from src into dst.
 * Both tensors must have identical shapes.
 */
void copy_tensor_values(const bt::Tensor &src, bt::Tensor &dst);

} // namespace bt::detail
