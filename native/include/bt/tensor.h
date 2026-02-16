#pragma once

#include <vector>

#include "bt/storage.h"

namespace bt {

class Tensor {
 public:
  std::shared_ptr<Storage> storage;
  int64_t storage_offset = 0;

  std::vector<int64_t> shape;
  std::vector<int64_t> strides;

  Tensor(const std::vector<int64_t>& shape);

  int dim() const;
  int64_t numel() const;
  bool is_contiguous() const;
  const float* data_ptr() const;
  float* data_ptr();

  Tensor operator+(const Tensor& t) const;
  Tensor operator+(const float value) const;
};

Tensor full(const std::vector<int64_t>& shape, float fill_value);
Tensor zeros(const std::vector<int64_t>& shape);
Tensor ones(const std::vector<int64_t>& shape);

}  // namespace bt