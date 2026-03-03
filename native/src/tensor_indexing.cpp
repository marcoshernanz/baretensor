/*
 * File: native/src/tensor_indexing.cpp
 * Purpose: Implements indexing-oriented tensor view operations and autograd.
 */

#include "bt/tensor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "bt/detail/autograd_record.h"
#include "bt/detail/dims.h"
#include "bt/detail/format.h"
#include "bt/detail/tensor_validation.h"

/*
 * Namespace: (anonymous)
 * Purpose: Private implementation details local to this translation unit.
 */
namespace {

/*
 * Struct: NormalizedSlice
 * Purpose: Stores a canonical positive-step slice over one dimension.
 */
struct NormalizedSlice {
  int64_t start = 0;
  int64_t stop = 0;
  int64_t step = 1;
  int64_t size = 0;
};

/*
 * Recursively copies data from a strided source layout into a strided
 * destination layout over a shared logical shape.
 */
void recursive_copy(size_t dim, size_t ndim, const std::vector<int64_t> &shape,
                    const float *src, float *dst,
                    const std::vector<int64_t> &src_strides,
                    const std::vector<int64_t> &dst_strides) {
  if (shape[dim] == 0) {
    return;
  }

  if (dim == ndim - 1) {
    for (int64_t i = 0; i < shape[dim]; ++i) {
      *dst = *src;
      src += src_strides[dim];
      dst += dst_strides[dim];
    }
    return;
  }

  for (int64_t i = 0; i < shape[dim]; ++i) {
    recursive_copy(dim + 1, ndim, shape, src, dst, src_strides, dst_strides);
    src += src_strides[dim];
    dst += dst_strides[dim];
  }
}

/*
 * Copies tensor values from src into dst assuming matching shapes.
 */
void copy_tensor_values(const bt::Tensor &src, bt::Tensor &dst) {
  if (src.shape != dst.shape) {
    throw std::runtime_error("copy_tensor_values failed: source and "
                             "destination tensor shapes do not match.");
  }

  if (src.ndim() == 0) {
    *dst.data_ptr() = *src.data_ptr();
    return;
  }

  recursive_copy(0, src.shape.size(), src.shape, src.data_ptr(), dst.data_ptr(),
                 src.strides, dst.strides);
}

/*
 * Normalizes and validates one integer index for select().
 */
[[nodiscard]] int64_t normalize_select_index_checked(
    const std::string_view operation_name, const std::vector<int64_t> &shape,
    const int64_t normalized_dim, const int64_t index) {
  const int64_t dim_size = shape[static_cast<size_t>(normalized_dim)];
  if (dim_size <= 0) {
    std::ostringstream oss;
    oss << operation_name << " failed for tensor with shape "
        << bt::detail::shape_to_string(shape)
        << ": cannot index into dimension " << normalized_dim << " with size "
        << dim_size << ".";
    throw std::invalid_argument(oss.str());
  }

  int64_t normalized_index = index;
  if (index < 0) {
    if (index < -dim_size) {
      std::ostringstream oss;
      oss << operation_name << " failed for tensor with shape "
          << bt::detail::shape_to_string(shape) << ": index " << index
          << " is out of range for dim " << normalized_dim << " with size "
          << dim_size << ".";
      throw std::invalid_argument(oss.str());
    }
    normalized_index = index + dim_size;
  }

  if (normalized_index < 0 || normalized_index >= dim_size) {
    std::ostringstream oss;
    oss << operation_name << " failed for tensor with shape "
        << bt::detail::shape_to_string(shape) << ": index " << index
        << " is out of range for dim " << normalized_dim << " with size "
        << dim_size << ".";
    throw std::invalid_argument(oss.str());
  }

  return normalized_index;
}

/*
 * Normalizes one slice bound against [0, dim_size] using Python-style rules
 * for positive steps.
 */
[[nodiscard]] int64_t normalize_slice_bound(const int64_t bound,
                                            const int64_t dim_size) {
  if (bound < 0) {
    if (bound < -dim_size) {
      return 0;
    }
    return bound + dim_size;
  }
  if (bound > dim_size) {
    return dim_size;
  }
  return bound;
}

/*
 * Normalizes and validates one positive-step slice for slice().
 */
[[nodiscard]] NormalizedSlice
normalize_slice_checked(const std::vector<int64_t> &shape,
                        const int64_t normalized_dim, const int64_t start,
                        const int64_t stop, const int64_t step) {
  if (step <= 0) {
    std::ostringstream oss;
    oss << "slice failed for tensor with shape "
        << bt::detail::shape_to_string(shape)
        << ": step must be greater than 0, got " << step << ".";
    throw std::invalid_argument(oss.str());
  }

  const int64_t dim_size = shape[static_cast<size_t>(normalized_dim)];
  const int64_t normalized_start = normalize_slice_bound(start, dim_size);
  const int64_t normalized_stop = normalize_slice_bound(stop, dim_size);

  int64_t normalized_size = 0;
  if (normalized_stop > normalized_start) {
    const int64_t span = normalized_stop - normalized_start;
    normalized_size = 1 + ((span - 1) / step);
  }

  return NormalizedSlice{
      .start = normalized_start,
      .stop = normalized_stop,
      .step = step,
      .size = normalized_size,
  };
}

/*
 * Class: SelectNode
 * Purpose: Backward scatter for Tensor::select(dim, index).
 */
class SelectNode final : public bt::Node {
public:
  SelectNode(const bt::Tensor &input, const int64_t dim, const int64_t index)
      : bt::Node({input}), input_shape_(input.shape), dim_(dim), index_(index) {
  }

  [[nodiscard]] std::vector<bt::Tensor>
  backward(const bt::Tensor &out_grad) const override {
    bt::Tensor input_grad = bt::zeros(input_shape_);
    if (out_grad.numel() == 0) {
      return {input_grad};
    }

    std::vector<int64_t> selected_shape = input_shape_;
    selected_shape.erase(selected_shape.begin() +
                         static_cast<std::ptrdiff_t>(dim_));

    std::vector<int64_t> selected_strides = input_grad.strides;
    selected_strides.erase(selected_strides.begin() +
                           static_cast<std::ptrdiff_t>(dim_));

    const int64_t selected_offset = index_ * input_grad.strides[dim_];
    bt::Tensor selected_grad(
        input_grad.storage, input_grad.storage_offset + selected_offset,
        std::move(selected_shape), std::move(selected_strides));
    copy_tensor_values(out_grad, selected_grad);
    return {input_grad};
  }

private:
  std::vector<int64_t> input_shape_;
  int64_t dim_ = 0;
  int64_t index_ = 0;
};

/*
 * Class: SliceNode
 * Purpose: Backward scatter for Tensor::slice(dim, start, stop, step).
 */
class SliceNode final : public bt::Node {
public:
  SliceNode(const bt::Tensor &input, const int64_t dim, const int64_t start,
            const int64_t step, const int64_t size)
      : bt::Node({input}), input_shape_(input.shape), dim_(dim), start_(start),
        step_(step), size_(size) {}

  [[nodiscard]] std::vector<bt::Tensor>
  backward(const bt::Tensor &out_grad) const override {
    bt::Tensor input_grad = bt::zeros(input_shape_);
    if (out_grad.numel() == 0) {
      return {input_grad};
    }

    std::vector<int64_t> sliced_shape = input_shape_;
    sliced_shape[static_cast<size_t>(dim_)] = size_;
    std::vector<int64_t> sliced_strides = input_grad.strides;
    sliced_strides[static_cast<size_t>(dim_)] *= step_;

    const int64_t sliced_offset = start_ * input_grad.strides[dim_];
    bt::Tensor sliced_grad(input_grad.storage,
                           input_grad.storage_offset + sliced_offset,
                           std::move(sliced_shape), std::move(sliced_strides));
    copy_tensor_values(out_grad, sliced_grad);
    return {input_grad};
  }

private:
  std::vector<int64_t> input_shape_;
  int64_t dim_ = 0;
  int64_t start_ = 0;
  int64_t step_ = 1;
  int64_t size_ = 0;
};

} // namespace

/*
 * Namespace: bt
 * Purpose: Public BareTensor C++ API surface.
 */
namespace bt {

/*
 * Returns a view that selects one index along dim, removing that dimension.
 */
Tensor Tensor::select(const int64_t dim, const int64_t index) const {
  bt::detail::validate_copy_metadata(*this, "select");
  const int64_t normalized_dim =
      detail::normalize_dim_checked("select", shape, dim, "dim");
  const int64_t normalized_index =
      normalize_select_index_checked("select", shape, normalized_dim, index);

  std::vector<int64_t> selected_shape = shape;
  selected_shape.erase(selected_shape.begin() +
                       static_cast<std::ptrdiff_t>(normalized_dim));
  std::vector<int64_t> selected_strides = strides;
  selected_strides.erase(selected_strides.begin() +
                         static_cast<std::ptrdiff_t>(normalized_dim));

  const int64_t selected_offset =
      storage_offset + (normalized_index * strides[normalized_dim]);
  Tensor out(storage, selected_offset, std::move(selected_shape),
             std::move(selected_strides));
  if (bt::detail::should_record_unary(*this)) {
    out.set_grad_fn(
        std::make_shared<SelectNode>(*this, normalized_dim, normalized_index));
  }
  return out;
}

/*
 * Returns a strided view sliced along dim over [start, stop) with step.
 */
Tensor Tensor::slice(const int64_t dim, const int64_t start, const int64_t stop,
                     const int64_t step) const {
  bt::detail::validate_copy_metadata(*this, "slice");
  const int64_t normalized_dim =
      detail::normalize_dim_checked("slice", shape, dim, "dim");
  const NormalizedSlice normalized_slice =
      normalize_slice_checked(shape, normalized_dim, start, stop, step);

  std::vector<int64_t> sliced_shape = shape;
  sliced_shape[static_cast<size_t>(normalized_dim)] = normalized_slice.size;
  std::vector<int64_t> sliced_strides = strides;
  sliced_strides[static_cast<size_t>(normalized_dim)] *= normalized_slice.step;

  const int64_t sliced_offset =
      storage_offset + (normalized_slice.start * strides[normalized_dim]);
  Tensor out(storage, sliced_offset, std::move(sliced_shape),
             std::move(sliced_strides));
  if (bt::detail::should_record_unary(*this)) {
    out.set_grad_fn(std::make_shared<SliceNode>(
        *this, normalized_dim, normalized_slice.start, normalized_slice.step,
        normalized_slice.size));
  }
  return out;
}

} // namespace bt
