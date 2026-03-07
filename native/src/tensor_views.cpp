/*
 * File: native/src/tensor_views.cpp
 * Purpose: Implements view/layout tensor ops and their autograd nodes.
 */

#include "bt/tensor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "bt/detail/autograd_record.h"
#include "bt/detail/dims.h"
#include "bt/detail/format.h"
#include "bt/detail/shape.h"
#include "bt/detail/tensor_copy.h"
#include "bt/detail/tensor_validation.h"

/*
 * Namespace: (anonymous)
 * Purpose: Private implementation details local to this translation unit.
 */
namespace {

/*
 * Class: ViewNode
 * Purpose: Backward mapping for view/reshape operations.
 */
class ViewNode final : public bt::Node {
public:
  explicit ViewNode(const bt::Tensor &input)
      : bt::Node({input}), input_shape_(input.shape) {}

  [[nodiscard]] std::vector<bt::Tensor>
  backward(const bt::Tensor &out_grad) const override {
    return {out_grad.reshape(input_shape_)};
  }

private:
  std::vector<int64_t> input_shape_;
};

/*
 * Class: PermuteNode
 * Purpose: Backward mapping for permutation operations.
 */
class PermuteNode final : public bt::Node {
public:
  PermuteNode(const bt::Tensor &input, const std::vector<int64_t> &inverse_dims)
      : bt::Node({input}), inverse_dims_(inverse_dims) {}

  [[nodiscard]] std::vector<bt::Tensor>
  backward(const bt::Tensor &out_grad) const override {
    return {out_grad.permute(inverse_dims_)};
  }

private:
  std::vector<int64_t> inverse_dims_;
};

/*
 * Class: ContiguousNode
 * Purpose: Backward pass-through for contiguous copies.
 */
class ContiguousNode final : public bt::Node {
public:
  explicit ContiguousNode(const bt::Tensor &input) : bt::Node({input}) {}

  [[nodiscard]] std::vector<bt::Tensor>
  backward(const bt::Tensor &out_grad) const override {
    return {out_grad};
  }
};

} // namespace

/*
 * Namespace: bt
 * Purpose: Public BareTensor C++ API surface.
 */
namespace bt {

/*
 * Returns a contiguous tensor with identical logical values and shape.
 * If the tensor is already contiguous, this returns an equivalent tensor
 * referencing the same storage.
 */
Tensor Tensor::contiguous() const {
  bt::detail::validate_copy_metadata(*this, "contiguous");

  if (is_contiguous()) {
    return *this;
  }

  Tensor out(shape);
  bt::detail::validate_copy_metadata(out, "contiguous");

  bt::detail::copy_tensor_values(*this, out);

  if (bt::detail::should_record_unary(*this)) {
    out.set_grad_fn(std::make_shared<ContiguousNode>(*this));
  }

  return out;
}

/*
 * Returns a view of this tensor with the requested shape when the current
 * shape and strides are layout-compatible with the target view.
 * Supports a single inferred '-1' dimension in the requested shape.
 */
Tensor Tensor::view(const std::vector<int64_t> &shape) const {
  bt::detail::validate_copy_metadata(*this, "view");

  std::vector<int64_t> target_shape = detail::infer_reshape_shape(this->shape, shape);
  std::optional<std::vector<int64_t>> target_strides =
      detail::infer_view_strides(this->shape, this->strides, target_shape);
  if (!target_strides.has_value()) {
    throw std::invalid_argument("Cannot view tensor with shape " +
                                detail::shape_to_string(this->shape) + " and strides " +
                                detail::shape_to_string(this->strides) + " as shape " +
                                detail::shape_to_string(target_shape) +
                                " without copying. Use contiguous() before view().");
  }

  Tensor out(storage, storage_offset, target_shape, *target_strides);
  if (bt::detail::should_record_unary(*this)) {
    out.set_grad_fn(std::make_shared<ViewNode>(*this));
  }
  return out;
}

/*
 * Returns a tensor with the requested shape, returning a view when possible
 * and otherwise returning a contiguous copy with the target shape.
 * Supports a single inferred '-1' dimension in the requested shape.
 */
Tensor Tensor::reshape(const std::vector<int64_t> &shape) const {
  bt::detail::validate_copy_metadata(*this, "reshape");

  std::vector<int64_t> target_shape = detail::infer_reshape_shape(this->shape, shape);
  std::optional<std::vector<int64_t>> target_strides =
      detail::infer_view_strides(this->shape, this->strides, target_shape);
  if (target_strides.has_value()) {
    Tensor out(storage, storage_offset, target_shape, *target_strides);
    if (bt::detail::should_record_unary(*this)) {
      out.set_grad_fn(std::make_shared<ViewNode>(*this));
    }
    return out;
  }

  return contiguous().view(target_shape);
}

/*
 * Returns a view with dimensions reordered according to dims.
 * Supports negative dimensions using Python-style indexing and requires
 * dims to be a full permutation of [0, ..., ndim()-1].
 */
Tensor Tensor::permute(const std::vector<int64_t> &dims) const {
  bt::detail::validate_copy_metadata(*this, "permute");

  const std::vector<int64_t> normalized_dims =
      detail::normalize_permutation_checked("permute", shape, dims);
  std::vector<int64_t> target_shape(shape.size(), 0);
  std::vector<int64_t> target_strides(strides.size(), 0);
  for (size_t i = 0; i < normalized_dims.size(); ++i) {
    const size_t source_dim = static_cast<size_t>(normalized_dims[i]);
    target_shape[i] = shape[source_dim];
    target_strides[i] = strides[source_dim];
  }

  Tensor out(storage, storage_offset, std::move(target_shape), std::move(target_strides));
  if (bt::detail::should_record_unary(*this)) {
    out.set_grad_fn(std::make_shared<PermuteNode>(
        *this, detail::invert_permutation(normalized_dims)));
  }
  return out;
}

/*
 * Returns a view with dim0 and dim1 swapped.
 * Supports negative dimensions using Python-style indexing.
 */
Tensor Tensor::transpose(const int64_t dim0, const int64_t dim1) const {
  bt::detail::validate_copy_metadata(*this, "transpose");

  const int64_t normalized_dim0 =
      detail::normalize_dim_checked("transpose", shape, dim0, "dim0");
  const int64_t normalized_dim1 =
      detail::normalize_dim_checked("transpose", shape, dim1, "dim1");
  if (normalized_dim0 == normalized_dim1) {
    return *this;
  }

  std::vector<int64_t> dims = detail::make_axis_order(shape.size());
  std::swap(dims[static_cast<size_t>(normalized_dim0)],
            dims[static_cast<size_t>(normalized_dim1)]);
  return permute(dims);
}

/*
 * Returns a 2-D matrix transpose view.
 * This operation requires ndim() == 2.
 */
Tensor Tensor::T() const {
  bt::detail::validate_copy_metadata(*this, "T");

  if (ndim() != 2) {
    std::ostringstream oss;
    oss << "T failed for tensor with shape " << detail::shape_to_string(shape)
        << ": expected ndim() == 2, but got " << ndim() << ".";
    throw std::invalid_argument(oss.str());
  }

  return permute({1, 0});
}

/*
 * Returns a view with the last two dimensions swapped.
 * Equivalent to transpose(-2, -1).
 */
Tensor Tensor::mT() const {
  bt::detail::validate_copy_metadata(*this, "mT");

  if (ndim() < 2) {
    std::ostringstream oss;
    oss << "mT failed for tensor with shape " << detail::shape_to_string(shape)
        << ": expected ndim() >= 2, but got " << ndim() << ".";
    throw std::invalid_argument(oss.str());
  }

  std::vector<int64_t> dims = detail::make_axis_order(shape.size());
  std::swap(dims[dims.size() - 2], dims[dims.size() - 1]);
  return permute(dims);
}

} /* namespace bt */
