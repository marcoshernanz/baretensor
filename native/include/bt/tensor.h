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
  Tensor(const std::vector<int64_t> &shape);

  /*
   * Constructs a tensor from owned data with the given shape.
   */
  Tensor(const std::vector<int64_t> &shape, std::vector<float> data);

  /*
   * Constructs a tensor view over existing storage and explicit metadata.
   * This constructor is intended for internal view-producing operations.
   */
  Tensor(const std::shared_ptr<Storage> storage, const int64_t storage_offset,
         const std::vector<int64_t> &shape,
         const std::vector<int64_t> &strides);

  /*
   * Returns the tensor rank.
   */
  [[nodiscard]] int ndim() const noexcept;

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
  [[nodiscard]] const float *data_ptr() const noexcept;

  /*
   * Returns a mutable data pointer at the tensor offset.
   */
  [[nodiscard]] float *data_ptr() noexcept;

  /*
   * Returns a contiguous tensor with identical logical values and shape.
   * If the tensor is already contiguous, this returns an equivalent tensor
   * referencing the same storage.
   */
  [[nodiscard]] Tensor contiguous() const;

  /*
   * Returns a view of this tensor with the requested shape when the current
   * shape and strides are layout-compatible with the target view.
   * Supports a single inferred '-1' dimension in the requested shape.
   */
  [[nodiscard]] Tensor view(const std::vector<int64_t> &shape) const;

  /*
   * Returns a tensor with the requested shape, returning a view when possible
   * and otherwise returning a contiguous copy with the target shape.
   * Supports a single inferred '-1' dimension in the requested shape.
   */
  [[nodiscard]] Tensor reshape(const std::vector<int64_t> &shape) const;

  /*
   * TODO
   */
  [[nodiscard]] Tensor permute(const std::vector<int> &dims);

  /*
   * Returns a view with dim0 and dim1 swapped.
   * Supports negative dimensions using Python-style indexing.
   */
  [[nodiscard]] Tensor transpose(int64_t dim0, int64_t dim1) const;

  /*
   * Returns a 2-D matrix transpose view.
   * This operation requires ndim() == 2.
   */
  [[nodiscard]] Tensor T() const;

  /*
   * Returns a view with the last two dimensions swapped.
   * Equivalent to transpose(-2, -1).
   */
  [[nodiscard]] Tensor mT() const;

  /*
   * Elementwise tensor-tensor addition.
   */
  Tensor operator+(const Tensor &t) const;

  /*
   * Elementwise tensor-scalar addition.
   */
  Tensor operator+(float rhs) const;

  /*
   * Elementwise tensor-tensor subtraction.
   */
  Tensor operator-(const Tensor &t) const;

  /*
   * Elementwise tensor-scalar subtraction.
   */
  Tensor operator-(float rhs) const;

  /*
   * Elementwise tensor-tensor multiplication.
   */
  Tensor operator*(const Tensor &t) const;

  /*
   * Elementwise tensor-scalar multiplication.
   */
  Tensor operator*(float rhs) const;

  /*
   * Elementwise tensor-tensor division.
   */
  Tensor operator/(const Tensor &t) const;

  /*
   * Elementwise tensor-scalar division.
   */
  Tensor operator/(float rhs) const;
};

/*
 * Creates a tensor filled with a constant value.
 */
[[nodiscard]] Tensor full(const std::vector<int64_t> &shape, float fill_value);

/*
 * Creates a tensor filled with zeros.
 */
[[nodiscard]] Tensor zeros(const std::vector<int64_t> &shape);

/*
 * Creates a tensor filled with ones.
 */
[[nodiscard]] Tensor ones(const std::vector<int64_t> &shape);

} /* namespace bt */
