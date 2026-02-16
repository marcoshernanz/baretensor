#pragma once

#include <vector>

#include "bt/storage.h"

namespace bt {

class Tensor {
 public:
  std::shared_ptr<Storage> storage;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;

  Tensor(const std::vector<int64_t>& shape);

  Tensor operator+(const Tensor& t) const;
  Tensor operator+(const float value) const;
};

Tensor full(const std::vector<int64_t>& shape, float fill_value);
Tensor zeros(const std::vector<int64_t>& shape);
Tensor ones(const std::vector<int64_t>& shape);

}  // namespace bt