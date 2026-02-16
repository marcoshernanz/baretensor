#pragma once

#include <vector>

#include "bt/storage.h"

namespace bt {

class Tensor {
 public:
  std::shared_ptr<Storage> data;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;

  Tensor(const std::vector<int64_t>& shape);
  void fill(float fill_value);

  std::string __repr__() const;

  std::vector<int64_t> stride() const;
  int64_t stride(int dim) const;
};

Tensor full(const std::vector<int64_t>& shape, float fill_value);
Tensor zeros(const std::vector<int64_t>& shape);
Tensor ones(const std::vector<int64_t>& shape);

}  // namespace bt