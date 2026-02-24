#pragma once

#include <cstdint>
#include <vector>

namespace bt {

class Storage {
 public:
  std::vector<float> data;

  explicit Storage(int64_t size);
  explicit Storage(const std::vector<float>& src);
  explicit Storage(std::vector<float>&& src) noexcept;

  void fill(float fill_value) noexcept;
};

}  // namespace bt
