/*
 * File: native/include/bt/detail/tensor_cast.h
 * Purpose: Declares shared dtype-conversion helpers for tensors.
 */

#pragma once

#include <cstdint>
#include <string_view>
#include <type_traits>

#include "bt/dtype.h"
#include "bt/tensor.h"

namespace bt::detail {

/*
 * Casts one scalar value between supported tensor dtypes.
 */
template <typename Src, typename Dst>
[[nodiscard]] inline Dst cast_tensor_scalar(const Src value, const std::string_view context) {
  if constexpr (std::is_same_v<Src, float> && std::is_same_v<Dst, int64_t>) {
    return checked_int64_from_double(static_cast<double>(value), context);
  } else {
    return static_cast<Dst>(value);
  }
}

/*
 * Throws when autograd is requested on a non-floating dtype.
 */
void ensure_grad_compatible_dtype(ScalarType dtype, std::string_view operation_name);

/*
 * Throws when autograd is requested on a non-floating tensor.
 */
inline void ensure_grad_compatible_dtype(const Tensor &tensor,
                                         const std::string_view operation_name) {
  ensure_grad_compatible_dtype(tensor.dtype(), operation_name);
}

/*
 * Converts tensor values to the requested dtype.
 * The output is always contiguous.
 */
[[nodiscard]] Tensor cast_tensor_dtype(const Tensor &tensor, ScalarType target_dtype,
                                       std::string_view context);

} // namespace bt::detail
