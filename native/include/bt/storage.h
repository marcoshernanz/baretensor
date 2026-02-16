#pragma once

#include <vector>

namespace bt {

class Storage {
 public:
  std::vector<float> data;

  Storage(int64_t size);

  void fill(float fill_value);
};

}  // namespace bt