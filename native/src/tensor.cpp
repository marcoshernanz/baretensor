#include "bt/tensor.h"

#include <array>
#include <stdexcept>
#include <vector>

Tensor full(std::vector<int> shape, float fill_value) {
  long long total_size = 1;
  for (int s : shape) {
    if (s < 0) throw std::runtime_error("Negative sizes are not allowed");
    total_size *= s;
  }

  Tensor tensor;
  tensor.data.resize(total_size, fill_value);
  tensor.shape = shape;
  return tensor;
}

Tensor zeros(std::vector<int> shape) { return full(shape, 0.0f); }

Tensor ones(std::vector<int> shape) { return full(shape, 1.0); }