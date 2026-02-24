#include "bt/storage.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace bt {

namespace {

[[nodiscard]] size_t checked_storage_size(int64_t size) {
  if (size < 0) throw std::invalid_argument("Storage size must be non-negative");
  return static_cast<size_t>(size);
}

}  // namespace

Storage::Storage(int64_t size) : data_(checked_storage_size(size)) {}

Storage::Storage(const std::vector<float>& src) : data_(src) {}

Storage::Storage(std::vector<float>&& src) noexcept : data_(std::move(src)) {}

const float* Storage::data_ptr() const noexcept { return data_.data(); }

float* Storage::data_ptr() noexcept { return data_.data(); }

size_t Storage::size() const noexcept { return data_.size(); }

void Storage::fill(float fill_value) noexcept {
  std::fill(data_.begin(), data_.end(), fill_value);
}

}  // namespace bt
