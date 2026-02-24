#include "bt/tensor.h"

#include <stdexcept>
#include <string>
#include <utility>

#include "bt/detail/format.h"
#include "bt/detail/shape.h"

namespace bt {

Tensor::Tensor(const std::vector<int64_t>& shape) : shape(shape) {
  const int64_t n = detail::checked_numel(shape);
  strides = detail::contiguous_strides(shape);
  storage = std::make_shared<Storage>(n);
}

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

int Tensor::dim() const noexcept { return static_cast<int>(shape.size()); }

int64_t Tensor::numel() const noexcept {
  int64_t n = 1;
  for (auto s : shape) {
    n *= s;
  }
  return n;
}

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

const float* Tensor::data_ptr() const noexcept {
  return storage->data_ptr() + storage_offset;
}

float* Tensor::data_ptr() noexcept {
  return storage->data_ptr() + storage_offset;
}

Tensor full(const std::vector<int64_t>& shape, float fill_value) {
  Tensor tensor(shape);
  tensor.storage->fill(fill_value);
  return tensor;
}

Tensor zeros(const std::vector<int64_t>& shape) { return full(shape, 0.0f); }

Tensor ones(const std::vector<int64_t>& shape) { return full(shape, 1.0f); }

}  // namespace bt
