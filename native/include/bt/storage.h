/*
 * File: native/include/bt/storage.h
 * Purpose: Declares the contiguous storage abstraction for typed tensors.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "bt/dtype.h"

/*
 * Namespace: bt
 * Purpose: Public BareTensor C++ API surface.
 */
namespace bt {

/*
 * Class: Storage
 * Purpose: Owns a contiguous typed buffer and basic memory operations.
 */
class Storage {
public:
  /*
   * Constructs storage with the requested number of elements and dtype.
   */
  explicit Storage(int64_t size, ScalarType dtype);

  ~Storage();

  Storage(const Storage &) = delete;
  Storage &operator=(const Storage &) = delete;
  Storage(Storage &&) = delete;
  Storage &operator=(Storage &&) = delete;

  /*
   * Returns the number of stored elements.
   */
  [[nodiscard]] size_t size() const noexcept;

  /*
   * Returns the number of bytes in the underlying buffer.
   */
  [[nodiscard]] size_t nbytes() const noexcept;

  /*
   * Returns the storage dtype.
   */
  [[nodiscard]] ScalarType dtype() const noexcept;

  /*
   * Returns a const pointer to the raw underlying bytes.
   */
  [[nodiscard]] const std::byte *raw_data() const noexcept;

  /*
   * Returns a mutable pointer to the raw underlying bytes.
   */
  [[nodiscard]] std::byte *raw_data() noexcept;

  /*
   * Returns a const typed data pointer after validating dtype agreement.
   */
  template <typename T> [[nodiscard]] const T *data_ptr() const;

  /*
   * Returns a mutable typed data pointer after validating dtype agreement.
   */
  template <typename T> [[nodiscard]] T *data_ptr();

  /*
   * Fills the entire buffer with a constant value.
   */
  void fill(double fill_value);

private:
  template <typename T> void validate_access_type() const;

  std::byte *data_ = nullptr;
  int64_t size_ = 0;
  ScalarType dtype_ = ScalarType::kFloat32;
};

template <typename T> const T *Storage::data_ptr() const {
  validate_access_type<T>();
  return reinterpret_cast<const T *>(data_);
}

template <typename T> T *Storage::data_ptr() {
  validate_access_type<T>();
  return reinterpret_cast<T *>(data_);
}

template <typename T> void Storage::validate_access_type() const {
  const ScalarType expected = scalar_type_of<T>();
  if (dtype_ != expected) {
    throw std::invalid_argument(std::string("Storage dtype mismatch: requested ") +
                                scalar_type_name(expected) + " but storage holds " +
                                scalar_type_name(dtype_) + ".");
  }
}

} /* namespace bt */
