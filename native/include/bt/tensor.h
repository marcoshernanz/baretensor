/*
 * File: native/include/bt/tensor.h
 * Purpose: Declares the tensor object and public tensor factory functions.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "bt/storage.h"

/*
 * Namespace: bt
 * Purpose: Public BareTensor C++ API surface.
 */
namespace bt {

/*
 * Class: Tensor
 * Purpose: Represents tensor metadata and storage-backed data access.
 */
class Tensor {
 public:
  /*
   * Shared backing storage for tensor values.
   */
  std::shared_ptr<Storage> storage;

  /*
   * Element offset from the start of storage.
   */
  int64_t storage_offset = 0;

  /*
   * Tensor shape in element counts per dimension.
   */
  std::vector<int64_t> shape;

  /*
   * Tensor strides in elements per dimension.
   */
  std::vector<int64_t> strides;

  /*
   * Constructs a tensor with allocated storage for the given shape.
   */
  Tensor(const std::vector<int64_t>& shape);

  /*
   * Constructs a tensor from owned data with the given shape.
   */
  Tensor(const std::vector<int64_t>& shape, std::vector<float> data);

  /*
   * Returns the tensor rank.
   */
  [[nodiscard]] int dim() const noexcept;

  /*
   * Returns the number of elements.
   */
  [[nodiscard]] int64_t numel() const noexcept;

  /*
   * Returns whether the tensor layout is contiguous.
   */
  [[nodiscard]] bool is_contiguous() const noexcept;

  /*
   * Returns a const data pointer at the tensor offset.
   */
  [[nodiscard]] const float* data_ptr() const noexcept;

  /*
   * Returns a mutable data pointer at the tensor offset.
   */
  [[nodiscard]] float* data_ptr() noexcept;

  /*
   * Elementwise tensor-tensor addition.
   */
  Tensor operator+(const Tensor& t) const;

  /*
   * Elementwise tensor-scalar addition.
   */
  Tensor operator+(float rhs) const;

  /*
   * Elementwise tensor-tensor subtraction.
   */
  Tensor operator-(const Tensor& t) const;

  /*
   * Elementwise tensor-scalar subtraction.
   */
  Tensor operator-(float rhs) const;

  /*
   * Elementwise tensor-tensor multiplication.
   */
  Tensor operator*(const Tensor& t) const;

  /*
   * Elementwise tensor-scalar multiplication.
   */
  Tensor operator*(float rhs) const;

  /*
   * Elementwise tensor-tensor division.
   */
  Tensor operator/(const Tensor& t) const;

  /*
   * Elementwise tensor-scalar division.
   */
  Tensor operator/(float rhs) const;
};

/*
 * Creates a tensor filled with a constant value.
 */
[[nodiscard]] Tensor full(const std::vector<int64_t>& shape, float fill_value);

/*
 * Creates a tensor filled with zeros.
 */
[[nodiscard]] Tensor zeros(const std::vector<int64_t>& shape);

/*
 * Creates a tensor filled with ones.
 */
[[nodiscard]] Tensor ones(const std::vector<int64_t>& shape);

} /* namespace bt */
