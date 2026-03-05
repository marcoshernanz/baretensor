/*
 * File: native/include/bt/storage.h
 * Purpose: Declares the contiguous float storage abstraction for tensors.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

/*
 * Namespace: bt
 * Purpose: Public BareTensor C++ API surface.
 */
namespace bt {

/*
 * Class: Storage
 * Purpose: Owns a contiguous float buffer and basic memory operations.
 */
class Storage {
public:
  /*
   * Constructs storage with the requested number of elements.
   */
  explicit Storage(int64_t size);

  /*
   * Constructs storage by copying from an existing vector.
   */
  explicit Storage(const std::vector<float> &src);

  /*
   * Constructs storage by moving from an existing vector.
   */
  explicit Storage(std::vector<float> &&src) noexcept;

  /*
   * Returns a const pointer to the underlying buffer.
   */
  [[nodiscard]] const float *data_ptr() const noexcept;

  /*
   * Returns a mutable pointer to the underlying buffer.
   */
  [[nodiscard]] float *data_ptr() noexcept;

  /*
   * Returns the number of stored elements.
   */
  [[nodiscard]] size_t size() const noexcept;

  /*
   * Fills the entire buffer with a constant value.
   */
  void fill(float fill_value) noexcept;

private:
  std::vector<float> data_;
};

} /* namespace bt */
