#pragma once

#include <cstdint>
#include <vector>

namespace bt {

class Storage {
 public:
  std::vector<float> data;

  explicit Storage(int64_t size);
  explicit Storage(std::vector<float>& data);

  void fill(float fill_value);
};

}  // namespace bt
