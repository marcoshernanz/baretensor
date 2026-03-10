/*
 * File: native/src/storage.cpp
 * Purpose: Implements the storage abstraction for contiguous typed buffers.
 */

#include "bt/storage.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <new>
#include <sstream>
#include <stdexcept>

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
[[nodiscard]] size_t checked_storage_size(const int64_t size) {
  if (size >= 0) {
    return static_cast<size_t>(size);
  }

  std::ostringstream oss;
  oss << "Invalid storage size " << size << ". Size must be non-negative.";
  throw std::invalid_argument(oss.str());
}

/*
 * Computes the total byte size for a typed storage allocation.
 */
[[nodiscard]] size_t checked_storage_nbytes(const int64_t size,
                                            const ScalarType dtype) {
  const size_t element_count = checked_storage_size(size);
  const size_t itemsize = scalar_type_itemsize(dtype);
  if (itemsize == 0) {
    throw std::invalid_argument("Invalid storage dtype.");
  }
  if (element_count == 0) {
    return 0;
  }
  if (element_count > (static_cast<size_t>(-1) / itemsize)) {
    throw std::overflow_error("Storage allocation size overflow.");
  }
  return element_count * itemsize;
}

/*
 * Allocates aligned raw bytes for typed tensor storage.
 */
[[nodiscard]] std::byte *allocate_storage_bytes(const size_t nbytes) {
  if (nbytes == 0) {
    return nullptr;
  }
  return static_cast<std::byte *>(
      ::operator new(nbytes, std::align_val_t(alignof(std::max_align_t))));
}

} // namespace

/*
 * Constructs storage with the requested number of elements and dtype.
 */
Storage::Storage(const int64_t size, const ScalarType dtype)
    : size_(static_cast<int64_t>(checked_storage_size(size))), dtype_(dtype) {
  data_ = allocate_storage_bytes(checked_storage_nbytes(size_, dtype_));
  if (data_ != nullptr) {
    std::memset(data_, 0, nbytes());
  }
}

/*
 * Releases any aligned storage bytes owned by this instance.
 */
Storage::~Storage() {
  if (data_ != nullptr) {
    ::operator delete(data_, std::align_val_t(alignof(std::max_align_t)));
  }
}

/*
 * Returns the number of stored elements.
 */
size_t Storage::size() const noexcept { return static_cast<size_t>(size_); }

/*
 * Returns the number of stored bytes.
 */
size_t Storage::nbytes() const noexcept {
  return static_cast<size_t>(size_) * scalar_type_itemsize(dtype_);
}

/*
 * Returns the storage dtype.
 */
ScalarType Storage::dtype() const noexcept { return dtype_; }

/*
 * Returns a const pointer to storage bytes.
 */
const std::byte *Storage::raw_data() const noexcept { return data_; }

/*
 * Returns a mutable pointer to storage bytes.
 */
std::byte *Storage::raw_data() noexcept { return data_; }

/*
 * Fills storage with a constant value in the storage dtype.
 */
void Storage::fill(const double fill_value) {
  visit_dtype(dtype_, [this, fill_value]<typename T>() {
    T typed_value{};
    if constexpr (std::is_same_v<T, int64_t>) {
      typed_value = checked_int64_from_double(fill_value, "Storage::fill()");
    } else {
      typed_value = static_cast<T>(fill_value);
    }
    T *data_ptr = this->data_ptr<T>();
    std::fill(data_ptr, data_ptr + size_, typed_value);
  });
}

} /* namespace bt */
