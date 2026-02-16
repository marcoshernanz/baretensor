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

Tensor full(const std::vector<int64_t>& shape, float fill_value) {
  Tensor tensor(shape);
  tensor.storage->fill(fill_value);
  return tensor;
}

Tensor zeros(const std::vector<int64_t>& shape) { return full(shape, 0.0f); }

Tensor ones(const std::vector<int64_t>& shape) { return full(shape, 1.0); }

}  // namespace bt