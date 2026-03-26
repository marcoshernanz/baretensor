/*
 * File: native/src/detail/tensor_cast.cpp
 * Purpose: Implements shared dtype-conversion helpers for tensors.
 */

#include "bt/detail/tensor_cast.h"

#include <cstdint>
#include <sstream>
#include <stdexcept>

namespace bt::detail {

/*
 * Throws when autograd is requested on a non-floating dtype.
 */
void ensure_grad_compatible_dtype(const ScalarType dtype, const std::string_view operation_name) {
  if (is_floating_point(dtype)) {
    return;
  }

  std::ostringstream oss;
  oss << operation_name << " is only supported for floating-point tensors, but got dtype "
      << scalar_type_name(dtype) << ".";
  throw std::invalid_argument(oss.str());
}

/*
 * Converts tensor values to the requested dtype.
 */
Tensor cast_tensor_dtype(const Tensor &tensor, const ScalarType target_dtype,
                         const std::string_view context) {
  if (tensor.dtype() == target_dtype) {
    return tensor;
  }

  const Tensor source = tensor.contiguous();
  Tensor out(tensor.shape, target_dtype);
  if (source.numel() == 0) {
    return out;
  }

  visit_dtype(source.dtype(), [&]<typename Src>() {
    visit_dtype(target_dtype, [&]<typename Dst>() {
      const Src *src_ptr = source.data_ptr<Src>();
      Dst *dst_ptr = out.data_ptr<Dst>();
      for (int64_t i = 0; i < source.numel(); ++i) {
        dst_ptr[i] = cast_tensor_scalar<Src, Dst>(src_ptr[i], context);
      }
    });
  });

  return out;
}

} // namespace bt::detail
