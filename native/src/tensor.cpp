#include "bt/tensor.h"

#include <array>
#include <stdexcept>
#include <vector>

namespace bt {
void Tensor::update_shape(std::vector<int64_t> shape) {
  this->shape = shape;
  size_t size = shape.size();

  int64_t num_elements = 1;
  for (auto s : shape) {
    if (s < 0) throw std::runtime_error("Negative sizes are not allowed");
    num_elements *= s;
  }
  data.resize(num_elements);

  stride.resize(size);
  stride[size - 1] = 1;
  for (int i = size - 2; i >= 0; i--) {
    stride[i] = stride[i + 1] * shape[i + 1];
  }
}

std::vector<int64_t> Tensor::stride() { return stride; }

int64_t Tensor::stride(int dim) {
  if (dim < 0 || dim >= stride.size()) {
    throw std::runtime_error("Invalid dimension");
  }

  return stride[dim];
}

Tensor full(std::vector<int64_t> shape, float fill_value) {
  Tensor tensor;
  tensor.update_shape(shape);
  return tensor;
}

Tensor zeros(std::vector<int64_t> shape) { return full(shape, 0.0f); }

Tensor ones(std::vector<int64_t> shape) { return full(shape, 1.0); }

}  // namespace bt