#include "bt/tensor.h"

#include <cstddef>
#include <stdexcept>

namespace bt {

Tensor::Tensor(const std::vector<int64_t>& shape) : shape(shape) {
  const size_t dim = shape.size();
  int64_t numel = 1;
  for (const int64_t s : shape) {
    if (s < 0) throw std::runtime_error("Negative sizes are not allowed");
    numel *= s;
  }

  strides.resize(dim);
  if (dim > 0) {
    strides[dim - 1] = 1;
    for (int i = dim - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
  }

  storage = std::make_shared<Storage>(numel);
}

Tensor::Tensor(const std::vector<int64_t>& shape,
               const std::vector<float>& data)
    : shape(shape) {
  const size_t dim = shape.size();
  int64_t numel = 1;
  for (const int64_t s : shape) {
    if (s < 0) throw std::runtime_error("Negative sizes are not allowed");
    numel *= s;
  }

  strides.resize(dim);
  if (dim > 0) {
    strides[dim - 1] = 1;
    for (int i = dim - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
  }

  storage = std::make_shared<Storage>(data);
}

int Tensor::dim() const { return static_cast<int>(shape.size()); }

int64_t Tensor::numel() const {
  int64_t n = 1;
  for (auto s : shape) {
    n *= s;
  }
  return n;
}

bool Tensor::is_contiguous() const {
  int64_t expected = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    if (shape[i] == 0) return true;
    if (shape[i] == 1) continue;
    if (strides[i] != expected) return false;
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

// Tensor tensor(const nb::ndarray<float>& array) {
//   std::vector<int64_t> shape(array.ndim());
//   std::vector<int64_t> strides(array.ndim());
//   for (int i = 0; i < array.ndim(); i++) {
//     shape[i] = array.shape(i);
//     strides[i] = array.stride(i);
//   }

//   Tensor tensor(shape);
//   tensor.strides = strides;
//   auto x = array.data();
//   tensor.storage->data = static_cast<float*>(array.data());
// }

}  // namespace bt
