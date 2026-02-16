#pragma once

#include <vector>

namespace bt {

class Tensor {
 private:
  std::vector<int64_t> strides;

 public:
  std::vector<float> data;
  std::vector<int64_t> shape;

  int64_t set_shape(std::vector<int64_t> shape);
};

Tensor full(std::vector<int64_t> shape, float fill_value);
Tensor zeros(std::vector<int64_t> shape);
Tensor ones(std::vector<int64_t> shape);

}  // namespace bt