#include "bt/tensor.h"

#include <stdexcept>
#include <vector>

namespace bt {
int64_t Tensor::set_shape(std::vector<int64_t> shape) {
  this->shape = shape;
  size_t size = shape.size();

  int64_t num_elements = 1;
  for (auto s : shape) {
    if (s < 0) throw std::runtime_error("Negative sizes are not allowed");
    num_elements *= s;
  }

  strides.resize(size);
  strides[size - 1] = 1;
  for (int i = size - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }

  return num_elements;
}

Tensor full(std::vector<int64_t> shape, float fill_value) {
  Tensor tensor;
  int64_t num_elements = tensor.set_shape(shape);
  tensor.data.resize(num_elements, fill_value);
  return tensor;
}

Tensor zeros(std::vector<int64_t> shape) { return full(shape, 0.0f); }

Tensor ones(std::vector<int64_t> shape) { return full(shape, 1.0); }

}  // namespace bt