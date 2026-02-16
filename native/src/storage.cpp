#include "bt/storage.h"

#include <algorithm>

namespace bt {

Storage::Storage(int64_t size) : data(size) {}

void Storage::fill(float fill_value) {
  std::fill(data.begin(), data.end(), fill_value);
}

}  // namespace bt