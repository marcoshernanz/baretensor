#pragma once

#include <vector>

namespace bt {

class Tensor {
 private:
  std::vector<float> data;
  std::vector<int64_t> strides;

 public:
  std::vector<int64_t> shape;

  void update_shape(std::vector<int64_t> shape);
};

Tensor full(std::vector<int64_t> shape, float fill_value);
Tensor zeros(std::vector<int64_t> shape);
Tensor ones(std::vector<int64_t> shape);

}  // namespace bt