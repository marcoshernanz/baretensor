#include "bt/tensor.h"

#include <array>
#include <stdexcept>
#include <vector>

namespace bt {
void Tensor::update_shape(std::vector<int64_t> shape) {
  this->shape = shape;
  size_t size = shape.size();

  data.resize(size);

  strides.resize(size);
  strides[strides.size() - 1] = 1;
  for (int i = size - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
}

Tensor full(std::vector<int64_t> shape, float fill_value) {
  int64_t total_size = 1;
  for (auto s : shape) {
    if (s < 0) throw std::runtime_error("Negative sizes are not allowed");
    total_size *= s;
  }

  Tensor tensor;
  tensor.update_shape(shape);
  return tensor;
}

Tensor zeros(std::vector<int64_t> shape) { return full(shape, 0.0f); }

Tensor ones(std::vector<int64_t> shape) { return full(shape, 1.0); }

}  // namespace bt