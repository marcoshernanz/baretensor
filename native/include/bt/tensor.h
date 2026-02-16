#pragma once

#include <vector>

class Tensor {
 private:
  std::vector<float> data;

 public:
  std::vector<int> shape;

  Tensor full(std::vector<int> size, float fill_value);
  Tensor zeros(std::vector<int> size);
  Tensor ones(std::vector<int> size);
};