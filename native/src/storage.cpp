/*
 * File: native/src/storage.cpp
 * Purpose: Implements the storage abstraction for contiguous tensor buffers.
 */

#include "bt/storage.h"

#include <algorithm>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <utility>

/*
 * Namespace: bt
 * Purpose: Public BareTensor C++ API surface.
 */
namespace bt {

/*
 * Namespace: (anonymous)
 * Purpose: Private implementation details local to this translation unit.
 */
namespace {

/*
 * Validates a storage element count and converts it to size_t.
 */
[[nodiscard]] size_t checked_storage_size(int64_t size) {
  if (size < 0) {
    std::ostringstream oss;
    oss << "Invalid storage size " << size << ": size must be non-negative.";
    throw std::invalid_argument(oss.str());
  }
  return static_cast<size_t>(size);
}

}  // namespace

/*
 * Constructs storage with the requested number of elements.
 */
Storage::Storage(int64_t size) : data_(checked_storage_size(size)) {}

/*
 * Constructs storage by copying an existing vector.
 */
Storage::Storage(const std::vector<float>& src) : data_(src) {}

/*
 * Constructs storage by moving an existing vector.
 */
Storage::Storage(std::vector<float>&& src) noexcept : data_(std::move(src)) {}

/*
 * Returns a const pointer to storage data.
 */
const float* Storage::data_ptr() const noexcept { return data_.data(); }

/*
 * Returns a mutable pointer to storage data.
 */
float* Storage::data_ptr() noexcept { return data_.data(); }

/*
 * Returns the number of stored elements.
 */
size_t Storage::size() const noexcept { return data_.size(); }

/*
 * Fills storage with a constant value.
 */
void Storage::fill(float fill_value) noexcept {
  std::fill(data_.begin(), data_.end(), fill_value);
}

} /* namespace bt */
