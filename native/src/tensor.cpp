/*
 * File: native/src/tensor.cpp
 * Purpose: Implements tensor construction, metadata queries, and factories.
 */

#include "bt/tensor.h"

#include <stdexcept>
#include <string>
#include <utility>

#include "bt/detail/broadcast.h"
#include "bt/detail/format.h"
#include "bt/detail/shape.h"

/*
 * Namespace: bt
 * Purpose: Public BareTensor C++ API surface.
 */
namespace bt {

/*
 * Constructs a tensor and allocates storage for the given shape.
 */
Tensor::Tensor(const std::vector<int64_t>& shape) : shape(shape) {
  const int64_t n = detail::checked_numel(shape);
  strides = detail::contiguous_strides(shape);
  storage = std::make_shared<Storage>(n);
}

/*
 * Constructs a tensor from provided shape and owned data vector.
 */
Tensor::Tensor(const std::vector<int64_t>& shape, std::vector<float> data)
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
 * TODO
 */
Tensor::Tensor(const std::shared_ptr<Storage> storage,
               const int64_t storage_offset, const std::vector<int64_t>& shape,
               const std::vector<int64_t>& strides)
    : storage(storage),
      storage_offset(storage_offset),
      shape(shape),
      strides(strides) {}

/*
 * Returns the tensor rank.
 */
int Tensor::dim() const noexcept { return static_cast<int>(shape.size()); }

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
    if (shape[i] == 0) return true;
    if (shape[i] == 1) continue;
    if (strides[i] != expected) return false;
    expected *= shape[i];
  }
  return true;
}

/*
 * Returns a const pointer to tensor data at storage offset.
 */
const float* Tensor::data_ptr() const noexcept {
  return storage->data_ptr() + storage_offset;
}

/*
 * Returns a mutable pointer to tensor data at storage offset.
 */
float* Tensor::data_ptr() noexcept {
  return storage->data_ptr() + storage_offset;
}

/*
 * TODO
 */
[[nodiscard]] Tensor Tensor::reshape(const std::vector<int64_t>& shape) {
  if (numel() != detail::checked_numel(shape)) {
    throw std::invalid_argument("invalid shape");
  }

  std::vector<int64_t> new_shape = detail::check_shape(self, shape);
  std::vector<int64_t> new_strides = detail::contiguous_strides(new_shape);

  return Tensor(storage, storage_offset, new_shape, new_strides);
}

/*
 * Creates a tensor filled with a constant value.
 */
Tensor full(const std::vector<int64_t>& shape, float fill_value) {
  Tensor tensor(shape);
  tensor.storage->fill(fill_value);
  return tensor;
}

/*
 * Creates a tensor filled with zeros.
 */
Tensor zeros(const std::vector<int64_t>& shape) { return full(shape, 0.0f); }

/*
 * Creates a tensor filled with ones.
 */
Tensor ones(const std::vector<int64_t>& shape) { return full(shape, 1.0f); }

} /* namespace bt */
