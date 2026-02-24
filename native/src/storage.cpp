#include "bt/storage.h"

#include <algorithm>
#include <utility>

namespace bt {

Storage::Storage(int64_t size) : data(size) {}

Storage::Storage(const std::vector<float>& src) : data(src) {}

Storage::Storage(std::vector<float>&& src) noexcept : data(std::move(src)) {}

void Storage::fill(float fill_value) noexcept {
  std::fill(data.begin(), data.end(), fill_value);
}

}  // namespace bt
