#pragma once

#include <vector>

namespace bt {

class Storage {
 private:
  std::vector<float> data;

 public:
  Storage(int64_t size);
  void fill(float fill_value);
};

}  // namespace bt