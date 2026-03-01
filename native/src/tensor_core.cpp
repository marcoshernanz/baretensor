/*
 * File: native/src/tensor_core.cpp
 * Purpose: Implements core tensor construction and autograd metadata APIs.
 */

#include "bt/tensor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bt/detail/format.h"
#include "bt/detail/shape.h"

/*
 * Namespace: bt
 * Purpose: Public BareTensor C++ API surface.
 */
namespace bt {

/*
 * Constructs a tensor and allocates storage for the given shape.
 */
Tensor::Tensor(const std::vector<int64_t> &shape) : shape(shape) {
  const int64_t n = detail::checked_numel(shape);
  strides = detail::contiguous_strides(shape);
  storage = std::make_shared<Storage>(n);
}

/*
 * Constructs a tensor from provided shape and owned data vector.
 */
Tensor::Tensor(const std::vector<int64_t> &shape, std::vector<float> data)
    : shape(shape) {
  const int64_t n = detail::checked_numel(shape);
  if (static_cast<int64_t>(data.size()) != n) {
    throw std::invalid_argument("Tensor data size mismatch for shape " +
                                detail::shape_to_string(shape) + ": expected " +
                                std::to_string(n) + " values but got " +
                                std::to_string(data.size()) + ".");
  }
  strides = detail::contiguous_strides(shape);
  storage = std::make_shared<Storage>(std::move(data));
}

/*
 * Constructs a tensor view over existing storage and explicit metadata.
 * This constructor is intended for internal view-producing operations.
 */
Tensor::Tensor(const std::shared_ptr<Storage> storage,
               const int64_t storage_offset, const std::vector<int64_t> &shape,
               const std::vector<int64_t> &strides)
    : storage(storage), storage_offset(storage_offset), shape(shape),
      strides(strides) {}

/*
 * Returns the tensor rank.
 */
int Tensor::ndim() const noexcept { return static_cast<int>(shape.size()); }

/*
 * Returns the total number of tensor elements.
 */
int64_t Tensor::numel() const noexcept {
  int64_t n = 1;
  for (const int64_t s : shape) {
    n *= s;
  }
  return n;
}

/*
 * Returns whether the current shape/stride metadata is contiguous.
 */
bool Tensor::is_contiguous() const noexcept {
  int64_t expected = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    if (shape[static_cast<size_t>(i)] == 0) {
      return true;
    }
    if (shape[static_cast<size_t>(i)] == 1) {
      continue;
    }
    if (strides[static_cast<size_t>(i)] != expected) {
      return false;
    }
    expected *= shape[static_cast<size_t>(i)];
  }
  return true;
}

/*
 * Returns a const pointer to tensor data at storage offset.
 */
const float *Tensor::data_ptr() const noexcept {
  return storage->data_ptr() + storage_offset;
}

/*
 * Returns a mutable pointer to tensor data at storage offset.
 */
float *Tensor::data_ptr() noexcept { return storage->data_ptr() + storage_offset; }

/*
 * Returns whether autograd is enabled for this tensor.
 */
bool Tensor::requires_grad() const noexcept {
  return autograd_meta != nullptr && autograd_meta->requires_grad;
}

/*
 * Sets whether this tensor tracks gradients in autograd.
 */
Tensor &Tensor::requires_grad_(const bool requires_grad) {
  if (!requires_grad) {
    if (autograd_meta != nullptr) {
      autograd_meta->requires_grad = false;
      autograd_meta->is_leaf = true;
      autograd_meta->grad = std::nullopt;
      autograd_meta->grad_fn = nullptr;
    }
    return *this;
  }

  if (autograd_meta == nullptr) {
    autograd_meta = std::make_shared<AutogradMeta>();
  }
  autograd_meta->requires_grad = true;
  if (autograd_meta->grad_fn == nullptr) {
    autograd_meta->is_leaf = true;
  }
  return *this;
}

/*
 * Returns whether this tensor is a leaf in the autograd graph.
 */
bool Tensor::is_leaf() const noexcept {
  if (autograd_meta == nullptr) {
    return true;
  }
  return autograd_meta->is_leaf;
}

/*
 * Returns the accumulated gradient for this tensor, if available.
 */
std::optional<Tensor> Tensor::grad() const {
  if (autograd_meta == nullptr) {
    return std::nullopt;
  }
  return autograd_meta->grad;
}

/*
 * Clears the accumulated gradient buffer for this tensor.
 */
void Tensor::zero_grad() {
  if (autograd_meta == nullptr) {
    return;
  }
  autograd_meta->grad = std::nullopt;
}

/*
 * Returns a tensor detached from autograd history.
 */
Tensor Tensor::detach() const {
  return Tensor(storage, storage_offset, shape, strides);
}

/*
 * Executes backward from this tensor.
 */
void Tensor::backward(const std::optional<Tensor> &gradient) const {
  autograd::backward(*this, gradient);
}

/*
 * Assigns a gradient function node to this tensor.
 */
void Tensor::set_grad_fn(const std::shared_ptr<Node> &grad_fn) {
  if (grad_fn == nullptr) {
    throw std::invalid_argument("set_grad_fn() expected a non-null node.");
  }
  if (autograd_meta == nullptr) {
    autograd_meta = std::make_shared<AutogradMeta>();
  }
  autograd_meta->requires_grad = true;
  autograd_meta->is_leaf = false;
  autograd_meta->grad_fn = grad_fn;
}

/*
 * Returns this tensor's gradient function node.
 */
std::shared_ptr<Node> Tensor::grad_fn() const {
  if (autograd_meta == nullptr) {
    return nullptr;
  }
  return autograd_meta->grad_fn;
}

/*
 * Accumulates a gradient tensor into this tensor's gradient buffer.
 */
void Tensor::accumulate_grad(const Tensor &incoming_grad) {
  if (incoming_grad.shape != shape) {
    std::ostringstream oss;
    oss << "accumulate_grad failed for tensor with shape "
        << detail::shape_to_string(shape) << ": gradient shape "
        << detail::shape_to_string(incoming_grad.shape)
        << " does not match tensor shape.";
    throw std::invalid_argument(oss.str());
  }

  if (autograd_meta == nullptr) {
    autograd_meta = std::make_shared<AutogradMeta>();
  }
  if (autograd_meta->grad.has_value()) {
    autograd::NoGradGuard guard;
    autograd_meta->grad = autograd_meta->grad.value() + incoming_grad;
    return;
  }

  autograd_meta->grad = incoming_grad.contiguous();
}

} /* namespace bt */
