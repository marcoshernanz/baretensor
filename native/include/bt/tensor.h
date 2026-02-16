#pragma once

#include <vector>

namespace bt {

class Tensor {
 public:
  std::vector<float> data;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;

  void update_strides();
};

Tensor full(std::vector<int64_t> shape, float fill_value);
Tensor zeros(std::vector<int64_t> shape);
Tensor ones(std::vector<int64_t> shape);

}  // namespace bt