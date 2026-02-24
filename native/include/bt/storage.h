#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bt {

class Storage {
 public:
  explicit Storage(int64_t size);
  explicit Storage(const std::vector<float>& src);
  explicit Storage(std::vector<float>&& src) noexcept;

  [[nodiscard]] const float* data_ptr() const noexcept;
  [[nodiscard]] float* data_ptr() noexcept;
  [[nodiscard]] size_t size() const noexcept;
  void fill(float fill_value) noexcept;

 private:
  std::vector<float> data_;
};

}  // namespace bt
