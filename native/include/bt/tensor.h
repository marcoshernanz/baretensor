#pragma once

#include <cstdint>
#include <memory>
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
  Tensor(const std::vector<int64_t>& shape, std::vector<float> data);

  [[nodiscard]] int dim() const noexcept;
  [[nodiscard]] int64_t numel() const noexcept;
  [[nodiscard]] bool is_contiguous() const noexcept;
  [[nodiscard]] const float* data_ptr() const noexcept;
  [[nodiscard]] float* data_ptr() noexcept;

  Tensor operator+(const Tensor& t) const;
  Tensor operator+(const float rhs) const;
  Tensor operator-(const Tensor& t) const;
  Tensor operator-(const float rhs) const;
  Tensor operator*(const Tensor& t) const;
  Tensor operator*(const float rhs) const;
  Tensor operator/(const Tensor& t) const;
  Tensor operator/(const float rhs) const;
};

Tensor full(const std::vector<int64_t>& shape, float fill_value);
Tensor zeros(const std::vector<int64_t>& shape);
Tensor ones(const std::vector<int64_t>& shape);

}  // namespace bt
