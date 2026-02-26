/*
 * File: native/src/tensor.cpp
 * Purpose: Implements tensor construction, metadata queries, and factories.
 */

#include "bt/tensor.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "bt/detail/format.h"
#include "bt/detail/shape.h"

/*
 * Namespace: (anonymous)
 * Purpose: Private implementation details local to this translation unit.
 */
namespace {

/*
 * Validates tensor metadata invariants required by low-level copy routines.
 */
void validate_copy_metadata(const bt::Tensor &tensor,
                            const std::string &operation_name) {
  if (!tensor.storage) {
    throw std::invalid_argument(operation_name +
                                " failed: tensor storage is null.");
  }

  if (tensor.storage_offset < 0) {
    throw std::invalid_argument(operation_name +
                                " failed for tensor with shape " +
                                bt::detail::shape_to_string(tensor.shape) +
                                ": storage offset must be non-negative, got " +
                                std::to_string(tensor.storage_offset) + ".");
  }

  if (tensor.shape.size() != tensor.strides.size()) {
    throw std::invalid_argument(
        operation_name + " failed for tensor with shape " +
        bt::detail::shape_to_string(tensor.shape) + ": shape rank " +
        std::to_string(tensor.shape.size()) + " does not match stride rank " +
        std::to_string(tensor.strides.size()) + ".");
  }
}

/*
 * Recursively copies data from a strided source layout into a strided
 * destination layout over a shared logical shape.
 */
void recursive_copy(size_t dim, size_t ndim, const std::vector<int64_t> &shape,
                    const float *src, float *dst,
                    const std::vector<int64_t> &src_strides,
                    const std::vector<int64_t> &dst_strides) {
  if (shape[dim] == 0)
    return;

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

} // namespace

/*
 * Namespace: bt
 * Purpose: Public BareTensor C++ API surface.
 */
namespace bt {

/*
 * Constructs a tensor and allocates storage for the given shape.
 */
Tensor::Tensor(const std::vector<int64_t> &shape) : shape(shape) {
  const int64_t n = detail::checked_numel(shape);
  strides = detail::contiguous_strides(shape);
  storage = std::make_shared<Storage>(n);
}

/*
 * Constructs a tensor from provided shape and owned data vector.
 */
Tensor::Tensor(const std::vector<int64_t> &shape, std::vector<float> data)
    : shape(shape) {
  const int64_t n = detail::checked_numel(shape);
  if (static_cast<int64_t>(data.size()) != n) {
    throw std::invalid_argument("Tensor data size mismatch for shape " +
                                detail::shape_to_string(shape) + ": expected " +
                                std::to_string(n) + " values but got " +
                                std::to_string(data.size()) + ".");
  }
  strides = detail::contiguous_strides(shape);
  storage = std::make_shared<Storage>(std::move(data));
}

/*
 * Constructs a tensor view over existing storage and explicit metadata.
 * This constructor is intended for internal view-producing operations.
 */
Tensor::Tensor(const std::shared_ptr<Storage> storage,
               const int64_t storage_offset, const std::vector<int64_t> &shape,
               const std::vector<int64_t> &strides)
    : storage(storage), storage_offset(storage_offset), shape(shape),
      strides(strides) {}

/*
 * Returns the tensor rank.
 */
int Tensor::ndim() const noexcept { return static_cast<int>(shape.size()); }

/*
 * Returns the total number of tensor elements.
 */
int64_t Tensor::numel() const noexcept {
  int64_t n = 1;
  for (auto s : shape) {
    n *= s;
  }
  return n;
}

/*
 * Returns whether the current shape/stride metadata is contiguous.
 */
bool Tensor::is_contiguous() const noexcept {
  int64_t expected = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    if (shape[i] == 0)
      return true;
    if (shape[i] == 1)
      continue;
    if (strides[i] != expected)
      return false;
    expected *= shape[i];
  }
  return true;
}

/*
 * Returns a const pointer to tensor data at storage offset.
 */
const float *Tensor::data_ptr() const noexcept {
  return storage->data_ptr() + storage_offset;
}

/*
 * Returns a mutable pointer to tensor data at storage offset.
 */
float *Tensor::data_ptr() noexcept {
  return storage->data_ptr() + storage_offset;
}

/*
 * Returns a contiguous tensor with identical logical values and shape.
 * If the tensor is already contiguous, this returns an equivalent tensor
 * referencing the same storage.
 */
Tensor Tensor::contiguous() const {
  validate_copy_metadata(*this, "contiguous");

  if (is_contiguous()) {
    return *this;
  }

  Tensor out(shape);
  validate_copy_metadata(out, "contiguous");

  const size_t ndim = shape.size();
  if (ndim == 0) {
    *out.data_ptr() = *data_ptr();
    return out;
  }

  recursive_copy(0, ndim, shape, data_ptr(), out.data_ptr(), strides,
                 out.strides);

  return out;
}

/*
 * Returns a view of this tensor with the requested shape when the current
 * shape and strides are layout-compatible with the target view.
 * Supports a single inferred '-1' dimension in the requested shape.
 */
Tensor Tensor::view(const std::vector<int64_t> &shape) const {
  validate_copy_metadata(*this, "view");

  std::vector<int64_t> target_shape =
      detail::infer_reshape_shape(this->shape, shape);
  std::optional<std::vector<int64_t>> target_strides =
      detail::infer_view_strides(this->shape, this->strides, target_shape);
  if (!target_strides.has_value()) {
    throw std::invalid_argument(
        "Cannot view tensor with shape " +
        detail::shape_to_string(this->shape) + " and strides " +
        detail::shape_to_string(this->strides) + " as shape " +
        detail::shape_to_string(target_shape) +
        " without copying. Use contiguous() before view().");
  }

  return Tensor(storage, storage_offset, target_shape, *target_strides);
}

/*
 * Returns a tensor with the requested shape, returning a view when possible
 * and otherwise returning a contiguous copy with the target shape.
 * Supports a single inferred '-1' dimension in the requested shape.
 */
Tensor Tensor::reshape(const std::vector<int64_t> &shape) const {
  validate_copy_metadata(*this, "reshape");

  std::vector<int64_t> target_shape =
      detail::infer_reshape_shape(this->shape, shape);
  std::optional<std::vector<int64_t>> target_strides =
      detail::infer_view_strides(this->shape, this->strides, target_shape);
  if (target_strides.has_value()) {
    return Tensor(storage, storage_offset, target_shape, *target_strides);
  }

  return contiguous().view(target_shape);
}

/*
 * TODO
 */
[[nodiscard]] Tensor Tensor::transpose(const int dim0, const int dim1) const {
  int real_dim0 = dim0 >= 0 ? dim0 : ndim() - dim0;
  int real_dim1 = dim1 >= 0 ? dim1 : ndim() - dim1;
  if (real_dim0 < 0 || real_dim0 >= ndim()) {
    throw std::invalid_argument("invalid dimension");
  } else if (real_dim1 < 0 || real_dim1 >= ndim()) {
    throw std::invalid_argument("invalid dimension");
  }

  std::vector<int64_t> target_shape(shape);
  std::vector<int64_t> target_strides(strides);
  std::swap(target_shape[real_dim0], target_shape[real_dim1]);
  std::swap(target_strides[real_dim0], target_strides[real_dim1]);

  return Tensor(storage, storage_offset, target_shape, target_strides);
}

/*
 * Creates a tensor filled with a constant value.
 */
Tensor full(const std::vector<int64_t> &shape, float fill_value) {
  Tensor tensor(shape);
  tensor.storage->fill(fill_value);
  return tensor;
}

/*
 * Creates a tensor filled with zeros.
 */
Tensor zeros(const std::vector<int64_t> &shape) { return full(shape, 0.0f); }

/*
 * Creates a tensor filled with ones.
 */
Tensor ones(const std::vector<int64_t> &shape) { return full(shape, 1.0f); }

} /* namespace bt */
