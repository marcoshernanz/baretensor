#pragma once

#include <vector>

namespace bt {

class Tensor {
 private:
  std::vector<float> data;

 public:
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;

  Tensor(const std::vector<int64_t>& shape);
  void fill(float fill_value);

  std::vector<int64_t> stride();
  int64_t stride(int dim);
};

Tensor full(const std::vector<int64_t>& shape, float fill_value);
Tensor zeros(const std::vector<int64_t>& shape);
Tensor ones(const std::vector<int64_t>& shape);

}  // namespace bt