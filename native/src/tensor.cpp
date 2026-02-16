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

  stride.resize(dim);
  if (dim > 0) {
    stride[dim - 1] = 1;
    for (int i = dim - 2; i >= 0; i--) {
      stride[i] = stride[i + 1] * shape[i + 1];
    }
  }

  storage = std::make_shared<Storage>(size);
}

int Tensor::dim() const { return shape.size(); }

int64_t Tensor::numel() const {
  int64_t n = 1;
  for (auto s : shape) {
    n *= s;
  }
  return n;
}

bool Tensor::is_contiguous() const {
  int64_t expected = 1;
  for (int i = shape.size() - 1; i >= 0; i--) {
    if (shape[i] == 0) return true;
    if (shape[i] == 1) continue;
    if (stride[i] != expected) return false;
    expected *= shape[i];
  }
  return true;
}

const float* Tensor::data_ptr() const {
  return storage->data.data() + storage_offset;
}

float* Tensor::data_ptr() { return storage->data.data() + storage_offset; }

Tensor full(const std::vector<int64_t>& shape, float fill_value) {
  Tensor tensor(shape);
  tensor.storage->fill(fill_value);
  return tensor;
}

Tensor zeros(const std::vector<int64_t>& shape) { return full(shape, 0.0f); }

Tensor ones(const std::vector<int64_t>& shape) { return full(shape, 1.0); }

}  // namespace bt