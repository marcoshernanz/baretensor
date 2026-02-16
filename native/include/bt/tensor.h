#pragma once

#include <vector>

class Tensor {
 public:
  std::vector<float> data;
  std::vector<int> shape;
};

// Tensor tensor(std::vector<float> data);
Tensor full(std::vector<int> shape, float fill_value);
Tensor zeros(std::vector<int> shape);
Tensor ones(std::vector<int> shape);