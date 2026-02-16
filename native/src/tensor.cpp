#include "bt/tensor.h"

#include <stdexcept>
#include <vector>

namespace bt {

Tensor::Tensor(const std::vector<int64_t>& shape) : shape(shape) {
  size_t dim = shape.size();

  int64_t num_elements = 1;
  for (auto s : shape) {
    if (s < 0) throw std::runtime_error("Negative sizes are not allowed");
    num_elements *= s;
  }

  strides.resize(dim);
  if (dim > 0) {
    strides[dim - 1] = 1;
    for (int i = dim - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
  }

  data.resize(num_elements);
}

void Tensor::fill(float fill_value) {
  std::fill(data.begin(), data.end(), fill_value);
}

std::string Tensor::__repr__() const { return "This is a test"; }

std::vector<int64_t> Tensor::stride() const { return strides; }

int64_t Tensor::stride(int dim) const {
  if (dim < 0) {
    dim = shape.size() + dim;
  }

  if (dim < 0 || dim >= strides.size()) {
    throw std::runtime_error("Invalid dimension");
  }

  return strides[dim];
}

Tensor full(const std::vector<int64_t>& shape, float fill_value) {
  Tensor tensor(shape);
  tensor.fill(fill_value);
  return tensor;
}

Tensor zeros(const std::vector<int64_t>& shape) { return full(shape, 0.0f); }

Tensor ones(const std::vector<int64_t>& shape) { return full(shape, 1.0); }

}  // namespace bt