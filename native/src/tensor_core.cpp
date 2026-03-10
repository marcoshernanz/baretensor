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
#include <type_traits>
#include <vector>

#include "bt/detail/format.h"
#include "bt/detail/shape.h"

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
 * Casts one element between supported tensor dtypes.
 */
template <typename Src, typename Dst>
[[nodiscard]] Dst cast_tensor_element(const Src value, const std::string_view context) {
  if constexpr (std::is_same_v<Src, float> && std::is_same_v<Dst, int64_t>) {
    return checked_int64_from_double(static_cast<double>(value), context);
  } else {
    return static_cast<Dst>(value);
  }
}

/*
 * Throws when autograd is requested on a non-floating tensor.
 */
void validate_requires_grad_dtype(const ScalarType dtype, const std::string_view operation_name) {
  if (is_floating_point(dtype)) {
    return;
  }

  std::ostringstream oss;
  oss << operation_name << " is only supported for floating-point tensors, but got dtype "
      << scalar_type_name(dtype) << ".";
  throw std::invalid_argument(oss.str());
}

} // namespace

/*
 * Constructs a tensor and allocates storage for the given shape and dtype.
 */
Tensor::Tensor(const std::vector<int64_t> &shape, const ScalarType dtype) : shape(shape) {
  const int64_t n = detail::checked_numel(shape);
  strides = detail::contiguous_strides(shape);
  storage = std::make_shared<Storage>(n, dtype);
}

/*
 * Constructs a tensor view over existing storage and explicit metadata.
 * This constructor is intended for internal view-producing operations.
 */
Tensor::Tensor(const std::shared_ptr<Storage> storage, const int64_t storage_offset,
               const std::vector<int64_t> &shape, const std::vector<int64_t> &strides)
    : storage(storage), storage_offset(storage_offset), shape(shape), strides(strides) {}

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
 * Returns the tensor dtype.
 */
ScalarType Tensor::dtype() const noexcept {
  return storage == nullptr ? ScalarType::kFloat32 : storage->dtype();
}

/*
 * Returns a const pointer to tensor bytes at storage offset.
 */
const std::byte *Tensor::raw_data_ptr() const noexcept {
  if (storage == nullptr) {
    return nullptr;
  }
  return storage->raw_data() + (storage_offset * scalar_type_itemsize(dtype()));
}

/*
 * Returns a mutable pointer to tensor bytes at storage offset.
 */
std::byte *Tensor::raw_data_ptr() noexcept {
  if (storage == nullptr) {
    return nullptr;
  }
  return storage->raw_data() + (storage_offset * scalar_type_itemsize(dtype()));
}

/*
 * Returns whether autograd is enabled for this tensor.
 */
bool Tensor::requires_grad() const noexcept {
  return autograd_meta != nullptr && autograd_meta->requires_grad;
}

/*
 * Sets whether this tensor tracks gradients in autograd.
 */
Tensor &Tensor::set_requires_grad(const bool requires_grad) {
  if (!requires_grad) {
    if (autograd_meta != nullptr) {
      autograd_meta->requires_grad = false;
      autograd_meta->is_leaf = true;
      autograd_meta->grad = std::nullopt;
      autograd_meta->grad_fn = nullptr;
    }
    return *this;
  }

  validate_requires_grad_dtype(dtype(), "set_requires_grad(true)");

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
Tensor Tensor::detach() const { return Tensor(storage, storage_offset, shape, strides); }

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

  validate_requires_grad_dtype(dtype(), "set_grad_fn()");

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
    oss << "accumulate_grad failed for tensor with shape " << detail::shape_to_string(shape)
        << ": gradient shape " << detail::shape_to_string(incoming_grad.shape)
        << " does not match tensor shape.";
    throw std::invalid_argument(oss.str());
  }

  validate_requires_grad_dtype(dtype(), "accumulate_grad()");
  if (incoming_grad.dtype() != dtype()) {
    std::ostringstream oss;
    oss << "accumulate_grad failed for tensor with dtype " << scalar_type_name(dtype())
        << ": gradient dtype " << scalar_type_name(incoming_grad.dtype()) << " does not match.";
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

/*
 * Returns a tensor converted to the requested dtype.
 */
Tensor Tensor::to(const ScalarType target_dtype) const {
  if (target_dtype == dtype()) {
    return *this;
  }

  if (requires_grad()) {
    throw std::invalid_argument(
        "to() does not support dtype conversion for tensors that require gradients.");
  }

  const Tensor source = contiguous();
  Tensor out(shape, target_dtype);
  if (source.numel() == 0) {
    return out;
  }

  visit_dtype(source.dtype(), [&]<typename Src>() {
    visit_dtype(target_dtype, [&]<typename Dst>() {
      const Src *src_ptr = source.data_ptr<Src>();
      Dst *dst_ptr = out.data_ptr<Dst>();
      for (int64_t i = 0; i < source.numel(); ++i) {
        dst_ptr[i] = cast_tensor_element<Src, Dst>(src_ptr[i], "to()");
      }
    });
  });

  return out;
}

} /* namespace bt */
