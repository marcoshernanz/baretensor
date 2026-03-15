/*
 * File: native/src/tensor_join.cpp
 * Purpose: Implements tensor join operations and their autograd nodes.
 */

#include "bt/tensor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "bt/detail/dims.h"
#include "bt/detail/format.h"
#include "bt/detail/tensor_copy.h"
#include "bt/detail/tensor_validation.h"

namespace {

[[nodiscard]] bool is_special_empty_cat_tensor(const bt::Tensor &tensor) {
  return tensor.ndim() == 1 && tensor.shape[0] == 0;
}

[[nodiscard]] std::string join_shapes_to_string(const std::vector<bt::Tensor> &tensors) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << bt::detail::shape_to_string(tensors[i].shape);
  }
  oss << "]";
  return oss.str();
}

void validate_join_metadata(const std::vector<bt::Tensor> &tensors,
                            const std::string_view operation_name) {
  for (const bt::Tensor &tensor : tensors) {
    bt::detail::validate_copy_metadata(tensor, std::string(operation_name));
  }
}

[[nodiscard]] bool should_record_join(const std::vector<bt::Tensor> &tensors) {
  if (!bt::autograd::is_grad_enabled()) {
    return false;
  }
  for (const bt::Tensor &tensor : tensors) {
    if (tensor.requires_grad()) {
      return true;
    }
  }
  return false;
}

class CatNode final : public bt::Node {
public:
  CatNode(const std::vector<bt::Tensor> &inputs, const int64_t dim) : bt::Node(inputs), dim_(dim) {}

  [[nodiscard]] std::vector<bt::Tensor> backward(const bt::Tensor &out_grad) const override {
    const std::vector<bt::Tensor> &inputs = this->inputs();
    std::vector<bt::Tensor> input_grads;
    input_grads.reserve(inputs.size());

    int64_t offset = 0;
    for (const bt::Tensor &input : inputs) {
      if (is_special_empty_cat_tensor(input)) {
        input_grads.push_back(bt::zeros(input.shape));
        continue;
      }

      const int64_t size = input.shape[static_cast<size_t>(dim_)];
      input_grads.push_back(out_grad.slice(dim_, offset, offset + size));
      offset += size;
    }
    return input_grads;
  }

private:
  int64_t dim_ = 0;
};

class StackNode final : public bt::Node {
public:
  StackNode(const std::vector<bt::Tensor> &inputs, const int64_t dim)
      : bt::Node(inputs), dim_(dim) {}

  [[nodiscard]] std::vector<bt::Tensor> backward(const bt::Tensor &out_grad) const override {
    const std::vector<bt::Tensor> &inputs = this->inputs();
    std::vector<bt::Tensor> input_grads;
    input_grads.reserve(inputs.size());

    for (size_t i = 0; i < inputs.size(); ++i) {
      input_grads.push_back(out_grad.select(dim_, static_cast<int64_t>(i)));
    }
    return input_grads;
  }

private:
  int64_t dim_ = 0;
};

} // namespace

namespace bt {

Tensor cat(const std::vector<Tensor> &tensors, const int64_t dim) {
  if (tensors.empty()) {
    throw std::invalid_argument("cat() expected a non-empty sequence of tensors.");
  }

  validate_join_metadata(tensors, "cat");

  const bool should_record = should_record_join(tensors);

  size_t reference_index = tensors.size();
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (!is_special_empty_cat_tensor(tensors[i])) {
      reference_index = i;
      break;
    }
  }

  if (reference_index == tensors.size()) {
    const bt::ScalarType dtype = tensors.front().dtype();
    for (size_t i = 1; i < tensors.size(); ++i) {
      if (tensors[i].dtype() != dtype) {
        std::ostringstream oss;
        oss << "cat failed for tensors with shapes " << join_shapes_to_string(tensors)
            << ": all tensors must have the same dtype, but found "
            << bt::scalar_type_name(dtype) << " and "
            << bt::scalar_type_name(tensors[i].dtype()) << ".";
        throw std::invalid_argument(oss.str());
      }
    }
    Tensor out = bt::zeros({0}, dtype);
    if (should_record) {
      out.set_grad_fn(std::make_shared<CatNode>(tensors, 0));
    }
    return out;
  }

  const Tensor &reference = tensors[reference_index];
  if (reference.ndim() == 0) {
    std::ostringstream oss;
    oss << "cat failed for tensors with shapes " << join_shapes_to_string(tensors)
        << ": zero-dimensional tensor at position " << reference_index
        << " cannot be concatenated.";
    throw std::invalid_argument(oss.str());
  }

  const int64_t normalized_dim =
      bt::detail::normalize_dim_checked("cat", reference.shape, dim, "dim");

  std::vector<int64_t> output_shape = reference.shape;
  int64_t concatenated_size = 0;
  for (size_t i = 0; i < tensors.size(); ++i) {
    const Tensor &tensor = tensors[i];
    if (is_special_empty_cat_tensor(tensor)) {
      continue;
    }
    if (tensor.ndim() == 0) {
      std::ostringstream oss;
      oss << "cat failed for tensors with shapes " << join_shapes_to_string(tensors)
          << ": zero-dimensional tensor at position " << i << " cannot be concatenated.";
      throw std::invalid_argument(oss.str());
    }
    if (tensor.ndim() != reference.ndim()) {
      std::ostringstream oss;
      oss << "cat failed for tensors with shapes " << join_shapes_to_string(tensors)
          << ": tensor at position " << i << " has rank " << tensor.ndim() << " but expected rank "
          << reference.ndim() << ".";
      throw std::invalid_argument(oss.str());
    }

    if (tensor.dtype() != reference.dtype()) {
      std::ostringstream oss;
      oss << "cat failed for tensors with shapes " << join_shapes_to_string(tensors)
          << ": tensor at position " << i << " has dtype "
          << bt::scalar_type_name(tensor.dtype()) << " but expected dtype "
          << bt::scalar_type_name(reference.dtype()) << ".";
      throw std::invalid_argument(oss.str());
    }

    for (int64_t axis = 0; axis < tensor.ndim(); ++axis) {
      if (axis == normalized_dim) {
        continue;
      }
      const int64_t actual = tensor.shape[static_cast<size_t>(axis)];
      const int64_t expected = reference.shape[static_cast<size_t>(axis)];
      if (actual != expected) {
        std::ostringstream oss;
        oss << "cat failed for tensors with shapes " << join_shapes_to_string(tensors)
            << ": sizes must match except in dimension " << normalized_dim << ". Expected size "
            << expected << " but got " << actual << " for tensor at position " << i << ".";
        throw std::invalid_argument(oss.str());
      }
    }

    concatenated_size += tensor.shape[static_cast<size_t>(normalized_dim)];
  }
  output_shape[static_cast<size_t>(normalized_dim)] = concatenated_size;

  Tensor out(output_shape, reference.dtype());
  int64_t offset = 0;
  for (const Tensor &tensor : tensors) {
    if (is_special_empty_cat_tensor(tensor)) {
      continue;
    }

    const int64_t size = tensor.shape[static_cast<size_t>(normalized_dim)];
    if (size != 0) {
      Tensor out_slice = out.slice(normalized_dim, offset, offset + size);
      bt::detail::copy_tensor_values(tensor, out_slice);
    }
    offset += size;
  }

  if (should_record) {
    out.set_grad_fn(std::make_shared<CatNode>(tensors, normalized_dim));
  }
  return out;
}

Tensor stack(const std::vector<Tensor> &tensors, const int64_t dim) {
  if (tensors.empty()) {
    throw std::invalid_argument("stack() expected a non-empty sequence of tensors.");
  }

  validate_join_metadata(tensors, "stack");

  const bool should_record = should_record_join(tensors);
  const Tensor &reference = tensors.front();
  const int64_t normalized_dim =
      bt::detail::normalize_insertion_dim_checked("stack", reference.shape, dim, "dim");

  for (size_t i = 1; i < tensors.size(); ++i) {
    const Tensor &tensor = tensors[i];
    if (tensor.dtype() != reference.dtype()) {
      std::ostringstream oss;
      oss << "stack failed for tensors with shapes " << join_shapes_to_string(tensors)
          << ": tensor at position " << i << " has dtype "
          << bt::scalar_type_name(tensor.dtype()) << " but expected dtype "
          << bt::scalar_type_name(reference.dtype()) << ".";
      throw std::invalid_argument(oss.str());
    }
    if (tensor.shape != reference.shape) {
      std::ostringstream oss;
      oss << "stack failed for tensors with shapes " << join_shapes_to_string(tensors)
          << ": tensor at position " << i << " has shape "
          << bt::detail::shape_to_string(tensor.shape) << " but expected shape "
          << bt::detail::shape_to_string(reference.shape) << ".";
      throw std::invalid_argument(oss.str());
    }
  }

  std::vector<int64_t> output_shape = reference.shape;
  output_shape.insert(output_shape.begin() + static_cast<std::ptrdiff_t>(normalized_dim),
                      static_cast<int64_t>(tensors.size()));

  Tensor out(output_shape, reference.dtype());
  for (size_t i = 0; i < tensors.size(); ++i) {
    Tensor out_slice = out.select(normalized_dim, static_cast<int64_t>(i));
    bt::detail::copy_tensor_values(tensors[i], out_slice);
  }

  if (should_record) {
    out.set_grad_fn(std::make_shared<StackNode>(tensors, normalized_dim));
  }
  return out;
}

} // namespace bt
