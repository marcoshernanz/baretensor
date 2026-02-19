#include "bt/storage.h"

#include <algorithm>

namespace bt {

Storage::Storage(int64_t size) : data(size) {}

Storage::Storage(std::vector<float>& data) { this->data = std::move(data); }

void Storage::fill(float fill_value) {
  std::fill(data.begin(), data.end(), fill_value);
}

}  // namespace bt