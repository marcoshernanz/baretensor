#include "bt/tensor.h"

#include <stdexcept>
#include <vector>

namespace bt {

Tensor::Tensor(const std::vector<int64_t>& shape) : shape(shape) {
  size_t dim = shape.size();

  int64_t size = 1;
  for (auto s : shape) {
    if (s < 0) throw std::runtime_error("Negative sizes are not allowed");
    size *= s;
  }

  strides.resize(dim);
  if (dim > 0) {
    strides[dim - 1] = 1;
    for (int i = dim - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
  }

  storage = std::make_shared<Storage>(size);
}

Tensor Tensor::operator+(const Tensor& t) const {
  if (shape != t.shape) {
    throw std::runtime_error("Tensors must have the same shape");
  }

  Tensor new_tensor(shape);
  int64_t size = storage->data.size();
  for (int64_t i = 0; i < size; i++) {
    new_tensor.storage->data[i] = storage->data[i] + t.storage->data[i];
  }

  return new_tensor;
}

Tensor Tensor::operator+(const float value) const {
  Tensor new_tensor(shape);
  int64_t size = storage->data.size();
  for (int64_t i = 0; i < size; i++) {
    new_tensor.storage->data[i] = storage->data[i] + value;
  }

  return new_tensor;
}

Tensor full(const std::vector<int64_t>& shape, float fill_value) {
  Tensor tensor(shape);
  tensor.storage->fill(fill_value);
  return tensor;
}

Tensor zeros(const std::vector<int64_t>& shape) { return full(shape, 0.0f); }

Tensor ones(const std::vector<int64_t>& shape) { return full(shape, 1.0); }

}  // namespace bt