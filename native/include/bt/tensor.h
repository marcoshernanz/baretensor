#pragma once

#include <vector>

namespace bt {

class Tensor {
 public:
  std::vector<float> data;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;

  int64_t set_shape(const std::vector<int64_t>& new_shape);
};

Tensor full(const std::vector<int64_t>& shape, float fill_value);
Tensor zeros(const std::vector<int64_t>& shape);
Tensor ones(const std::vector<int64_t>& shape);

}  // namespace bt