/*
 * File: native/src/tensor.cpp
 * Purpose: Implements tensor construction, metadata queries, and factories.
 */

#include "bt/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "bt/detail/broadcast.h"
#include "bt/detail/dims.h"
#include "bt/detail/format.h"
#include "bt/detail/shape.h"

/*
 * Namespace: (anonymous)
 * Purpose: Private implementation details local to this translation unit.
 */
namespace {

/*
 * Validates tensor metadata invariants required by low-level copy routines.
 */
void validate_copy_metadata(const bt::Tensor &tensor,
                            const std::string &operation_name) {
  if (!tensor.storage) {
    throw std::invalid_argument(operation_name +
                                " failed: tensor storage is null.");
  }

  if (tensor.storage_offset < 0) {
    throw std::invalid_argument(operation_name +
                                " failed for tensor with shape " +
                                bt::detail::shape_to_string(tensor.shape) +
                                ": storage offset must be non-negative, got " +
                                std::to_string(tensor.storage_offset) + ".");
  }

  if (tensor.shape.size() != tensor.strides.size()) {
    throw std::invalid_argument(
        operation_name + " failed for tensor with shape " +
        bt::detail::shape_to_string(tensor.shape) + ": shape rank " +
        std::to_string(tensor.shape.size()) + " does not match stride rank " +
        std::to_string(tensor.strides.size()) + ".");
  }
}

/*
 * Recursively copies data from a strided source layout into a strided
 * destination layout over a shared logical shape.
 */
void recursive_copy(size_t dim, size_t ndim, const std::vector<int64_t> &shape,
                    const float *src, float *dst,
                    const std::vector<int64_t> &src_strides,
                    const std::vector<int64_t> &dst_strides) {
  if (shape[dim] == 0)
    return;

  if (dim == ndim - 1) {
    for (int64_t i = 0; i < shape[dim]; ++i) {
      *dst = *src;
      src += src_strides[dim];
      dst += dst_strides[dim];
    }
    return;
  }

  for (int64_t i = 0; i < shape[dim]; ++i) {
    recursive_copy(dim + 1, ndim, shape, src, dst, src_strides, dst_strides);
    src += src_strides[dim];
    dst += dst_strides[dim];
  }
}

/*
 * Builds an identity permutation [0, 1, ..., rank - 1].
 */
std::vector<int64_t> make_axis_order(const size_t rank) {
  std::vector<int64_t> dims(rank, 0);
  for (size_t i = 0; i < rank; ++i) {
    dims[i] = static_cast<int64_t>(i);
  }
  return dims;
}

/*
 * Stores canonical matmul metadata for one operand.
 */
struct MatmulCanonicalInput {
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  bool was_1d = false;
};

/*
 * Kernel parameters for one matrix-multiply invocation.
 */
struct MatmulKernelParams {
  int64_t m = 0;
  int64_t k = 0;
  int64_t n = 0;
  int64_t lhs_m_stride = 0;
  int64_t lhs_k_stride = 0;
  int64_t rhs_k_stride = 0;
  int64_t rhs_n_stride = 0;
  int64_t out_m_stride = 0;
  int64_t out_n_stride = 0;
};

/*
 * Builds a shape-pair string for diagnostics.
 */
[[nodiscard]] std::string matmul_shapes_to_string(const bt::Tensor &lhs,
                                                  const bt::Tensor &rhs) {
  std::ostringstream oss;
  oss << "shapes " << bt::detail::shape_to_string(lhs.shape) << " and "
      << bt::detail::shape_to_string(rhs.shape);
  return oss.str();
}

/*
 * Canonicalizes an input for matmul execution.
 */
[[nodiscard]] MatmulCanonicalInput
canonicalize_matmul_input(const bt::Tensor &tensor, const bool prepend_for_1d) {
  if (tensor.ndim() != 1) {
    return MatmulCanonicalInput{tensor.shape, tensor.strides, false};
  }

  if (prepend_for_1d) {
    return MatmulCanonicalInput{
        {1, tensor.shape[0]}, {0, tensor.strides[0]}, true};
  }

  return MatmulCanonicalInput{
      {tensor.shape[0], 1}, {tensor.strides[0], 0}, true};
}

/*
 * Removes temporary singleton dimensions introduced by 1-D promotion.
 */
[[nodiscard]] std::vector<int64_t>
matmul_result_shape(const std::vector<int64_t> &full_shape,
                    const bool lhs_was_1d, const bool rhs_was_1d) {
  std::vector<int64_t> out_shape = full_shape;
  if (lhs_was_1d) {
    out_shape.erase(out_shape.end() - 2);
  }
  if (rhs_was_1d) {
    out_shape.erase(out_shape.end() - 1);
  }
  return out_shape;
}

/*
 * Computes one strided matrix multiplication:
 *   out[m, n] = lhs[m, k] @ rhs[k, n]
 */
void matmul_one_matrix(const float *lhs, const float *rhs, float *out,
                       const MatmulKernelParams &params) {
  for (int64_t row = 0; row < params.m; ++row) {
    const float *lhs_row_ptr = lhs + (row * params.lhs_m_stride);
    float *out_row_ptr = out + (row * params.out_m_stride);
    for (int64_t col = 0; col < params.n; ++col) {
      float acc = 0.0f;
      const float *lhs_k_ptr = lhs_row_ptr;
      const float *rhs_k_ptr = rhs + (col * params.rhs_n_stride);
      for (int64_t kk = 0; kk < params.k; ++kk) {
        acc += (*lhs_k_ptr) * (*rhs_k_ptr);
        lhs_k_ptr += params.lhs_k_stride;
        rhs_k_ptr += params.rhs_k_stride;
      }
      out_row_ptr[col * params.out_n_stride] = acc;
    }
  }
}

/*
 * Recursively applies matrix multiplication over broadcasted batch dimensions.
 */
void recursive_batched_matmul(const size_t dim,
                              const std::vector<int64_t> &batch_shape,
                              const float *lhs, const float *rhs, float *out,
                              const std::vector<int64_t> &lhs_batch_strides,
                              const std::vector<int64_t> &rhs_batch_strides,
                              const std::vector<int64_t> &out_batch_strides,
                              const MatmulKernelParams &params) {
  if (dim == batch_shape.size()) {
    matmul_one_matrix(lhs, rhs, out, params);
    return;
  }

  if (batch_shape[dim] == 0) {
    return;
  }

  for (int64_t i = 0; i < batch_shape[dim]; ++i) {
    recursive_batched_matmul(dim + 1, batch_shape, lhs, rhs, out,
                             lhs_batch_strides, rhs_batch_strides,
                             out_batch_strides, params);
    lhs += lhs_batch_strides[dim];
    rhs += rhs_batch_strides[dim];
    out += out_batch_strides[dim];
  }
}

/*
 * Stores precomputed metadata for a reduction.
 */
struct ReductionPlan {
  std::vector<bool> reduce_mask;
  std::vector<int64_t> input_to_output_dim;
  std::vector<int64_t> output_shape;
};

/*
 * Normalizes one reduction dimension using Python-style indexing.
 */
[[nodiscard]] int64_t
normalize_reduction_dim(const bt::Tensor &tensor, const int64_t dim,
                        const size_t dim_index,
                        const std::string_view operation_name) {
  const int64_t rank = static_cast<int64_t>(tensor.shape.size());
  const int64_t normalized = dim < 0 ? dim + rank : dim;
  if (normalized >= 0 && normalized < rank) {
    return normalized;
  }

  std::ostringstream oss;
  oss << operation_name << " failed for tensor with shape "
      << bt::detail::shape_to_string(tensor.shape) << ": dim[" << dim_index
      << "]=" << dim << " is out of range for rank " << rank << ".";
  throw std::invalid_argument(oss.str());
}

/*
 * Normalizes and validates reduction dimensions.
 */
[[nodiscard]] std::vector<int64_t>
normalize_reduction_dims(const bt::Tensor &tensor,
                         const std::vector<int64_t> &dim,
                         const std::string_view operation_name) {
  const int64_t rank = static_cast<int64_t>(tensor.shape.size());
  std::vector<int64_t> normalized_dims;
  normalized_dims.reserve(dim.size());

  std::vector<bool> seen(static_cast<size_t>(rank), false);
  for (size_t i = 0; i < dim.size(); ++i) {
    const int64_t normalized =
        normalize_reduction_dim(tensor, dim[i], i, operation_name);
    if (seen[static_cast<size_t>(normalized)]) {
      std::ostringstream oss;
      oss << operation_name << " failed for tensor with shape "
          << bt::detail::shape_to_string(tensor.shape) << ": dimension "
          << normalized << " appears more than once in dim.";
      throw std::invalid_argument(oss.str());
    }
    seen[static_cast<size_t>(normalized)] = true;
    normalized_dims.push_back(normalized);
  }

  return normalized_dims;
}

/*
 * Builds reduction metadata for reduction(dim, keepdim).
 */
[[nodiscard]] ReductionPlan
build_reduction_plan(const bt::Tensor &tensor,
                     const std::vector<int64_t> &normalized_dims,
                     const bool keepdim) {
  const size_t rank = tensor.shape.size();
  ReductionPlan plan{
      .reduce_mask = std::vector<bool>(rank, false),
      .input_to_output_dim = std::vector<int64_t>(rank, -1),
      .output_shape = {},
  };

  for (const int64_t reduced_dim : normalized_dims) {
    plan.reduce_mask[static_cast<size_t>(reduced_dim)] = true;
  }

  if (keepdim) {
    plan.output_shape = tensor.shape;
    for (size_t dim = 0; dim < rank; ++dim) {
      if (plan.reduce_mask[dim]) {
        plan.output_shape[dim] = 1;
      } else {
        plan.input_to_output_dim[dim] = static_cast<int64_t>(dim);
      }
    }
    return plan;
  }

  plan.output_shape.reserve(rank - normalized_dims.size());
  int64_t out_dim = 0;
  for (size_t dim = 0; dim < rank; ++dim) {
    if (plan.reduce_mask[dim]) {
      continue;
    }
    plan.output_shape.push_back(tensor.shape[dim]);
    plan.input_to_output_dim[dim] = out_dim;
    ++out_dim;
  }

  return plan;
}

/*
 * Recursively accumulates input values into an output tensor for reduction.
 */
void recursive_sum_reduce(const size_t dim, const std::vector<int64_t> &shape,
                          const std::vector<int64_t> &input_strides,
                          const std::vector<int64_t> &output_strides,
                          const std::vector<int64_t> &input_to_output_dim,
                          const float *input_ptr, float *output_ptr) {
  if (dim == shape.size()) {
    *output_ptr += *input_ptr;
    return;
  }

  if (shape[dim] == 0) {
    return;
  }

  const int64_t out_dim = input_to_output_dim[dim];
  for (int64_t index = 0; index < shape[dim]; ++index) {
    const float *next_input_ptr = input_ptr + (index * input_strides[dim]);
    float *next_output_ptr = output_ptr;
    if (out_dim >= 0) {
      next_output_ptr += index * output_strides[static_cast<size_t>(out_dim)];
    }

    recursive_sum_reduce(dim + 1, shape, input_strides, output_strides,
                         input_to_output_dim, next_input_ptr, next_output_ptr);
  }
}

/*
 * Executes additive reduction with a precomputed plan.
 */
[[nodiscard]] bt::Tensor sum_with_plan(const bt::Tensor &tensor,
                                       const ReductionPlan &plan) {
  bt::Tensor out(plan.output_shape);
  out.storage->fill(0.0f);

  recursive_sum_reduce(0, tensor.shape, tensor.strides, out.strides,
                       plan.input_to_output_dim, tensor.data_ptr(),
                       out.data_ptr());
  return out;
}

/*
 * Computes the number of input elements reduced into each output element.
 */
[[nodiscard]] int64_t
reduction_element_count(const bt::Tensor &tensor,
                        const std::vector<int64_t> &normalized_dims) {
  int64_t count = 1;
  for (const int64_t dim : normalized_dims) {
    count *= tensor.shape[static_cast<size_t>(dim)];
  }
  return count;
}

/*
 * Recursively propagates maximum values into an output tensor for reduction.
 */
void recursive_max_reduce(const size_t dim, const std::vector<int64_t> &shape,
                          const std::vector<int64_t> &input_strides,
                          const std::vector<int64_t> &output_strides,
                          const std::vector<int64_t> &input_to_output_dim,
                          const float *input_ptr, float *output_ptr) {
  if (dim == shape.size()) {
    if (*input_ptr > *output_ptr) {
      *output_ptr = *input_ptr;
    }
    return;
  }

  if (shape[dim] == 0) {
    return;
  }

  const int64_t out_dim = input_to_output_dim[dim];
  for (int64_t index = 0; index < shape[dim]; ++index) {
    const float *next_input_ptr = input_ptr + (index * input_strides[dim]);
    float *next_output_ptr = output_ptr;
    if (out_dim >= 0) {
      next_output_ptr += index * output_strides[static_cast<size_t>(out_dim)];
    }

    recursive_max_reduce(dim + 1, shape, input_strides, output_strides,
                         input_to_output_dim, next_input_ptr, next_output_ptr);
  }
}

/*
 * Validates that the requested reduction does not reduce over zero elements.
 */
void validate_non_empty_reduction(const bt::Tensor &tensor,
                                  const std::vector<int64_t> &normalized_dims,
                                  const std::string_view operation_name) {
  if (reduction_element_count(tensor, normalized_dims) != 0) {
    return;
  }

  std::ostringstream oss;
  oss << operation_name << " failed for tensor with shape "
      << bt::detail::shape_to_string(tensor.shape) << " and dim "
      << bt::detail::shape_to_string(normalized_dims)
      << ": cannot perform reduction over zero elements.";
  throw std::invalid_argument(oss.str());
}

/*
 * Executes max reduction with a precomputed plan.
 */
[[nodiscard]] bt::Tensor max_with_plan(const bt::Tensor &tensor,
                                       const ReductionPlan &plan) {
  bt::Tensor out(plan.output_shape);
  out.storage->fill(-std::numeric_limits<float>::infinity());

  recursive_max_reduce(0, tensor.shape, tensor.strides, out.strides,
                       plan.input_to_output_dim, tensor.data_ptr(),
                       out.data_ptr());
  return out;
}

[[nodiscard]] bool should_record_unary(const bt::Tensor &input) {
  return bt::autograd::is_grad_enabled() && input.requires_grad();
}

[[nodiscard]] bool should_record_binary(const bt::Tensor &lhs,
                                        const bt::Tensor &rhs) {
  return bt::autograd::is_grad_enabled() &&
         (lhs.requires_grad() || rhs.requires_grad());
}

void throw_autograd_not_implemented(const std::string_view op_name) {
  std::ostringstream oss;
  oss << "Autograd support for " << op_name << " is not implemented yet.";
  throw std::runtime_error(oss.str());
}

[[nodiscard]] std::vector<int64_t>
invert_permutation(const std::vector<int64_t> &dims) {
  std::vector<int64_t> inverse(dims.size(), 0);
  for (size_t i = 0; i < dims.size(); ++i) {
    inverse[static_cast<size_t>(dims[i])] = static_cast<int64_t>(i);
  }
  return inverse;
}

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

class SumNode final : public bt::Node {
public:
  SumNode(const bt::Tensor &input, const std::vector<int64_t> &reduced_dims,
          const bool keepdim)
      : bt::Node({input}), input_shape_(input.shape), keepdim_(keepdim) {
    reduced_dims_ = reduced_dims;
    std::sort(reduced_dims_.begin(), reduced_dims_.end());
  }

  [[nodiscard]] std::vector<bt::Tensor>
  backward(const bt::Tensor &out_grad) const override {
    bt::Tensor grad = out_grad;
    if (!keepdim_) {
      std::vector<int64_t> reshape_shape = grad.shape;
      for (const int64_t dim : reduced_dims_) {
        reshape_shape.insert(reshape_shape.begin() + dim, 1);
      }
      grad = grad.reshape(reshape_shape);
    }

    const bt::Tensor expanded = grad * bt::ones(input_shape_);
    return {expanded};
  }

private:
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> reduced_dims_;
  bool keepdim_ = false;
};

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

class ContiguousNode final : public bt::Node {
public:
  explicit ContiguousNode(const bt::Tensor &input) : bt::Node({input}) {}

  [[nodiscard]] std::vector<bt::Tensor>
  backward(const bt::Tensor &out_grad) const override {
    return {out_grad};
  }
};

class MatmulNode final : public bt::Node {
public:
  MatmulNode(const bt::Tensor &lhs, const bt::Tensor &rhs)
      : bt::Node({lhs, rhs}) {}

  [[nodiscard]] std::vector<bt::Tensor>
  backward(const bt::Tensor &out_grad) const override {
    const std::vector<bt::Tensor> &inputs = this->inputs();
    const bt::Tensor &lhs = inputs[0];
    const bt::Tensor &rhs = inputs[1];

    const MatmulCanonicalInput lhs_canonical_meta =
        canonicalize_matmul_input(lhs, true);
    const MatmulCanonicalInput rhs_canonical_meta =
        canonicalize_matmul_input(rhs, false);

    const bt::Tensor lhs_canonical(lhs.storage, lhs.storage_offset,
                                   lhs_canonical_meta.shape,
                                   lhs_canonical_meta.strides);
    const bt::Tensor rhs_canonical(rhs.storage, rhs.storage_offset,
                                   rhs_canonical_meta.shape,
                                   rhs_canonical_meta.strides);

    const std::vector<int64_t> lhs_batch_shape(
        lhs_canonical_meta.shape.begin(), lhs_canonical_meta.shape.end() - 2);
    const std::vector<int64_t> rhs_batch_shape(
        rhs_canonical_meta.shape.begin(), rhs_canonical_meta.shape.end() - 2);
    const std::vector<int64_t> batch_shape =
        bt::detail::infer_broadcast_shape(lhs_batch_shape, rhs_batch_shape);

    std::vector<int64_t> full_out_shape = batch_shape;
    full_out_shape.push_back(
        lhs_canonical_meta.shape[lhs_canonical_meta.shape.size() - 2]);
    full_out_shape.push_back(
        rhs_canonical_meta.shape[rhs_canonical_meta.shape.size() - 1]);

    const bt::Tensor out_grad_canonical = out_grad.reshape(full_out_shape);

    bt::Tensor lhs_grad_canonical =
        out_grad_canonical.matmul(rhs_canonical.mT());
    bt::Tensor rhs_grad_canonical =
        lhs_canonical.mT().matmul(out_grad_canonical);

    bt::Tensor lhs_grad = bt::autograd::reduce_sum_to_shape(
        lhs_grad_canonical, lhs_canonical_meta.shape);
    bt::Tensor rhs_grad = bt::autograd::reduce_sum_to_shape(
        rhs_grad_canonical, rhs_canonical_meta.shape);

    if (lhs_canonical_meta.was_1d) {
      lhs_grad = lhs_grad.reshape(lhs.shape);
    }
    if (rhs_canonical_meta.was_1d) {
      rhs_grad = rhs_grad.reshape(rhs.shape);
    }

    return {lhs_grad, rhs_grad};
  }
};

} // namespace

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
  for (auto s : shape) {
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
    if (shape[i] == 0)
      return true;
    if (shape[i] == 1)
      continue;
    if (strides[i] != expected)
      return false;
    expected *= shape[i];
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
float *Tensor::data_ptr() noexcept {
  return storage->data_ptr() + storage_offset;
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

/*
 * Returns a contiguous tensor with identical logical values and shape.
 * If the tensor is already contiguous, this returns an equivalent tensor
 * referencing the same storage.
 */
Tensor Tensor::contiguous() const {
  validate_copy_metadata(*this, "contiguous");

  if (is_contiguous()) {
    return *this;
  }

  Tensor out(shape);
  validate_copy_metadata(out, "contiguous");

  const size_t ndim = shape.size();
  if (ndim == 0) {
    *out.data_ptr() = *data_ptr();
    return out;
  }

  recursive_copy(0, ndim, shape, data_ptr(), out.data_ptr(), strides,
                 out.strides);

  if (should_record_unary(*this)) {
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
  validate_copy_metadata(*this, "view");

  std::vector<int64_t> target_shape =
      detail::infer_reshape_shape(this->shape, shape);
  std::optional<std::vector<int64_t>> target_strides =
      detail::infer_view_strides(this->shape, this->strides, target_shape);
  if (!target_strides.has_value()) {
    throw std::invalid_argument(
        "Cannot view tensor with shape " +
        detail::shape_to_string(this->shape) + " and strides " +
        detail::shape_to_string(this->strides) + " as shape " +
        detail::shape_to_string(target_shape) +
        " without copying. Use contiguous() before view().");
  }

  Tensor out(storage, storage_offset, target_shape, *target_strides);
  if (should_record_unary(*this)) {
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
  validate_copy_metadata(*this, "reshape");

  std::vector<int64_t> target_shape =
      detail::infer_reshape_shape(this->shape, shape);
  std::optional<std::vector<int64_t>> target_strides =
      detail::infer_view_strides(this->shape, this->strides, target_shape);
  if (target_strides.has_value()) {
    Tensor out(storage, storage_offset, target_shape, *target_strides);
    if (should_record_unary(*this)) {
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
  validate_copy_metadata(*this, "permute");

  const std::vector<int64_t> normalized_dims =
      detail::normalize_permutation_checked("permute", shape, dims);
  std::vector<int64_t> target_shape(shape.size(), 0);
  std::vector<int64_t> target_strides(strides.size(), 0);
  for (size_t i = 0; i < normalized_dims.size(); ++i) {
    const size_t source_dim = static_cast<size_t>(normalized_dims[i]);
    target_shape[i] = shape[source_dim];
    target_strides[i] = strides[source_dim];
  }

  Tensor out(storage, storage_offset, std::move(target_shape),
             std::move(target_strides));
  if (should_record_unary(*this)) {
    out.set_grad_fn(std::make_shared<PermuteNode>(
        *this, invert_permutation(normalized_dims)));
  }
  return out;
}

/*
 * Returns a view with dim0 and dim1 swapped.
 * Supports negative dimensions using Python-style indexing.
 */
Tensor Tensor::transpose(const int64_t dim0, const int64_t dim1) const {
  validate_copy_metadata(*this, "transpose");

  const int64_t normalized_dim0 =
      detail::normalize_dim_checked("transpose", shape, dim0, "dim0");
  const int64_t normalized_dim1 =
      detail::normalize_dim_checked("transpose", shape, dim1, "dim1");
  if (normalized_dim0 == normalized_dim1) {
    return *this;
  }

  std::vector<int64_t> dims = make_axis_order(shape.size());
  std::swap(dims[static_cast<size_t>(normalized_dim0)],
            dims[static_cast<size_t>(normalized_dim1)]);
  return permute(dims);
}

/*
 * Returns a 2-D matrix transpose view.
 * This operation requires ndim() == 2.
 */
Tensor Tensor::T() const {
  validate_copy_metadata(*this, "T");

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
  validate_copy_metadata(*this, "mT");

  if (ndim() < 2) {
    std::ostringstream oss;
    oss << "mT failed for tensor with shape " << detail::shape_to_string(shape)
        << ": expected ndim() >= 2, but got " << ndim() << ".";
    throw std::invalid_argument(oss.str());
  }

  std::vector<int64_t> dims = make_axis_order(shape.size());
  std::swap(dims[dims.size() - 2], dims[dims.size() - 1]);
  return permute(dims);
}

/*
 * Returns the matrix product of this tensor and tensor2 using matmul
 * semantics equivalent to PyTorch for dense tensors.
 */
Tensor Tensor::matmul(const Tensor &tensor2) const {
  validate_copy_metadata(*this, "matmul");
  validate_copy_metadata(tensor2, "matmul");

  if (ndim() == 0 || tensor2.ndim() == 0) {
    std::ostringstream oss;
    oss << "matmul failed for tensors with "
        << matmul_shapes_to_string(*this, tensor2)
        << ": both tensors must be at least 1-D, but got " << ndim()
        << "-D and " << tensor2.ndim() << "-D.";
    throw std::invalid_argument(oss.str());
  }

  const MatmulCanonicalInput lhs = canonicalize_matmul_input(*this, true);
  const MatmulCanonicalInput rhs = canonicalize_matmul_input(tensor2, false);

  const int64_t lhs_k = lhs.shape[lhs.shape.size() - 1];
  const int64_t rhs_k = rhs.shape[rhs.shape.size() - 2];
  if (lhs_k != rhs_k) {
    std::ostringstream oss;
    oss << "matmul failed for tensors with "
        << matmul_shapes_to_string(*this, tensor2)
        << ": inner dimensions must match (lhs.shape[-1] == rhs.shape[-2]), "
           "got "
        << lhs_k << " and " << rhs_k << ".";
    throw std::invalid_argument(oss.str());
  }

  const std::vector<int64_t> lhs_batch_shape(lhs.shape.begin(),
                                             lhs.shape.end() - 2);
  const std::vector<int64_t> rhs_batch_shape(rhs.shape.begin(),
                                             rhs.shape.end() - 2);

  std::vector<int64_t> batch_shape;
  try {
    batch_shape =
        detail::infer_broadcast_shape(lhs_batch_shape, rhs_batch_shape);
  } catch (const std::invalid_argument &err) {
    std::ostringstream oss;
    oss << "matmul failed for tensors with "
        << matmul_shapes_to_string(*this, tensor2)
        << ": batch dimensions are not broadcastable: " << err.what();
    throw std::invalid_argument(oss.str());
  }

  const std::vector<int64_t> lhs_batch_strides(lhs.strides.begin(),
                                               lhs.strides.end() - 2);
  const std::vector<int64_t> rhs_batch_strides(rhs.strides.begin(),
                                               rhs.strides.end() - 2);

  const std::vector<int64_t> lhs_batch_broadcast_strides =
      detail::aligned_broadcast_strides(lhs_batch_shape, lhs_batch_strides,
                                        batch_shape);
  const std::vector<int64_t> rhs_batch_broadcast_strides =
      detail::aligned_broadcast_strides(rhs_batch_shape, rhs_batch_strides,
                                        batch_shape);

  std::vector<int64_t> full_out_shape = batch_shape;
  full_out_shape.push_back(lhs.shape[lhs.shape.size() - 2]);
  full_out_shape.push_back(rhs.shape[rhs.shape.size() - 1]);

  Tensor out_full(full_out_shape);
  if (out_full.numel() != 0) {
    const std::vector<int64_t> out_batch_strides(out_full.strides.begin(),
                                                 out_full.strides.end() - 2);
    const MatmulKernelParams params{
        .m = lhs.shape[lhs.shape.size() - 2],
        .k = lhs_k,
        .n = rhs.shape[rhs.shape.size() - 1],
        .lhs_m_stride = lhs.strides[lhs.strides.size() - 2],
        .lhs_k_stride = lhs.strides[lhs.strides.size() - 1],
        .rhs_k_stride = rhs.strides[rhs.strides.size() - 2],
        .rhs_n_stride = rhs.strides[rhs.strides.size() - 1],
        .out_m_stride = out_full.strides[out_full.strides.size() - 2],
        .out_n_stride = out_full.strides[out_full.strides.size() - 1],
    };

    recursive_batched_matmul(0, batch_shape, data_ptr(), tensor2.data_ptr(),
                             out_full.data_ptr(), lhs_batch_broadcast_strides,
                             rhs_batch_broadcast_strides, out_batch_strides,
                             params);
  }

  const std::vector<int64_t> out_shape =
      matmul_result_shape(full_out_shape, lhs.was_1d, rhs.was_1d);
  Tensor out = out_full.reshape(out_shape);
  if (should_record_binary(*this, tensor2)) {
    out.set_grad_fn(std::make_shared<MatmulNode>(*this, tensor2));
  }
  return out;
}

/*
 * Returns the sum of all tensor elements as a scalar tensor.
 */
Tensor Tensor::sum() const { return sum(make_axis_order(shape.size()), false); }

/*
 * Returns the sum reduced along one dimension.
 */
Tensor Tensor::sum(const int64_t dim, const bool keepdim) const {
  return sum(std::vector<int64_t>{dim}, keepdim);
}

/*
 * Returns the sum reduced along one or more dimensions.
 */
Tensor Tensor::sum(const std::vector<int64_t> &dim, const bool keepdim) const {
  validate_copy_metadata(*this, "sum");

  const std::vector<int64_t> normalized_dims =
      normalize_reduction_dims(*this, dim, "sum");
  const ReductionPlan plan =
      build_reduction_plan(*this, normalized_dims, keepdim);
  Tensor out = sum_with_plan(*this, plan);
  if (should_record_unary(*this)) {
    out.set_grad_fn(std::make_shared<SumNode>(*this, normalized_dims, keepdim));
  }
  return out;
}

/*
 * Returns the mean of all tensor elements as a scalar tensor.
 */
Tensor Tensor::mean() const {
  return mean(make_axis_order(shape.size()), false);
}

/*
 * Returns the mean reduced along one dimension.
 */
Tensor Tensor::mean(const int64_t dim, const bool keepdim) const {
  return mean(std::vector<int64_t>{dim}, keepdim);
}

/*
 * Returns the mean reduced along one or more dimensions.
 */
Tensor Tensor::mean(const std::vector<int64_t> &dim, const bool keepdim) const {
  validate_copy_metadata(*this, "mean");
  if (should_record_unary(*this)) {
    throw_autograd_not_implemented("mean");
  }

  const std::vector<int64_t> normalized_dims =
      normalize_reduction_dims(*this, dim, "mean");
  const ReductionPlan plan =
      build_reduction_plan(*this, normalized_dims, keepdim);
  const Tensor reduced_sum = sum_with_plan(*this, plan);
  const int64_t reduced_element_count =
      reduction_element_count(*this, normalized_dims);
  return reduced_sum / static_cast<float>(reduced_element_count);
}

/*
 * Returns the maximum of all tensor elements as a scalar tensor.
 */
Tensor Tensor::max() const { return max(make_axis_order(shape.size()), false); }

/*
 * Returns the maximum reduced along one dimension.
 */
Tensor Tensor::max(const int64_t dim, const bool keepdim) const {
  return max(std::vector<int64_t>{dim}, keepdim);
}

/*
 * Returns the maximum reduced along one or more dimensions.
 */
Tensor Tensor::max(const std::vector<int64_t> &dim, const bool keepdim) const {
  validate_copy_metadata(*this, "max");
  if (should_record_unary(*this)) {
    throw_autograd_not_implemented("max");
  }

  const std::vector<int64_t> normalized_dims =
      normalize_reduction_dims(*this, dim, "max");
  validate_non_empty_reduction(*this, normalized_dims, "max");
  const ReductionPlan plan =
      build_reduction_plan(*this, normalized_dims, keepdim);
  return max_with_plan(*this, plan);
}

/*
 * Returns a tensor containing softmax values computed along dim.
 */
Tensor Tensor::softmax(const int64_t dim) const {
  validate_copy_metadata(*this, "softmax");
  if (should_record_unary(*this)) {
    throw_autograd_not_implemented("softmax");
  }

  const int64_t normalized_dim =
      detail::normalize_dim_checked("softmax", shape, dim, "dim");
  if (shape[static_cast<size_t>(normalized_dim)] == 0) {
    return exp();
  }

  const Tensor max_values = max(normalized_dim, true);
  const Tensor shifted = (*this) - max_values;
  const Tensor exp_values = shifted.exp();
  const Tensor normalizer = exp_values.sum(normalized_dim, true);
  return exp_values / normalizer;
}

/*
 * Returns a tensor containing log-softmax values computed along dim.
 */
Tensor Tensor::log_softmax(const int64_t dim) const {
  validate_copy_metadata(*this, "log_softmax");
  if (should_record_unary(*this)) {
    throw_autograd_not_implemented("log_softmax");
  }

  const int64_t normalized_dim =
      detail::normalize_dim_checked("log_softmax", shape, dim, "dim");
  if (shape[static_cast<size_t>(normalized_dim)] == 0) {
    return log();
  }

  const Tensor max_values = max(normalized_dim, true);
  const Tensor shifted = (*this) - max_values;
  const Tensor log_normalizer = shifted.exp().sum(normalized_dim, true).log();
  return shifted - log_normalizer;
}

/*
 * Applies layer normalization over the trailing normalized_shape dimensions.
 */
Tensor layer_norm(const Tensor &input,
                  const std::vector<int64_t> &normalized_shape,
                  const std::optional<Tensor> &weight,
                  const std::optional<Tensor> &bias, const float eps) {
  validate_copy_metadata(input, "layer_norm");
  if (should_record_unary(input) ||
      (weight.has_value() && should_record_unary(*weight)) ||
      (bias.has_value() && should_record_unary(*bias))) {
    throw_autograd_not_implemented("layer_norm");
  }
  if (weight.has_value()) {
    validate_copy_metadata(*weight, "layer_norm");
  }
  if (bias.has_value()) {
    validate_copy_metadata(*bias, "layer_norm");
  }

  const auto make_error_prefix = [&input, &normalized_shape]() {
    return std::string("layer_norm failed for input shape ") +
           detail::shape_to_string(input.shape) + " and normalized_shape " +
           detail::shape_to_string(normalized_shape) + ": ";
  };

  if (normalized_shape.empty()) {
    throw std::invalid_argument(
        make_error_prefix() +
        "normalized_shape must contain at least one dimension.");
  }

  for (size_t dim = 0; dim < normalized_shape.size(); ++dim) {
    if (normalized_shape[dim] > 0) {
      continue;
    }

    std::ostringstream oss;
    oss << make_error_prefix() << "normalized_shape[" << dim
        << "] must be positive, got " << normalized_shape[dim] << ".";
    throw std::invalid_argument(oss.str());
  }

  if (!std::isfinite(eps) || eps <= 0.0f) {
    std::ostringstream oss;
    oss << make_error_prefix() << "eps must be a finite value > 0, got " << eps
        << ".";
    throw std::invalid_argument(oss.str());
  }

  const size_t normalized_rank = normalized_shape.size();
  const size_t input_rank = input.shape.size();
  if (normalized_rank > input_rank) {
    std::ostringstream oss;
    oss << make_error_prefix()
        << "normalized_shape rank must be <= input rank, got "
        << normalized_rank << " and " << input_rank << ".";
    throw std::invalid_argument(oss.str());
  }

  const size_t tail_start = input_rank - normalized_rank;
  for (size_t dim = 0; dim < normalized_rank; ++dim) {
    const int64_t input_tail_dim = input.shape[tail_start + dim];
    if (input_tail_dim == normalized_shape[dim]) {
      continue;
    }

    std::ostringstream oss;
    oss << make_error_prefix() << "input tail dimensions "
        << detail::shape_to_string(std::vector<int64_t>(
               input.shape.begin() + tail_start, input.shape.end()))
        << " must match normalized_shape.";
    throw std::invalid_argument(oss.str());
  }

  if (weight.has_value() && weight->shape != normalized_shape) {
    std::ostringstream oss;
    oss << make_error_prefix() << "weight shape "
        << detail::shape_to_string(weight->shape)
        << " must match normalized_shape.";
    throw std::invalid_argument(oss.str());
  }
  if (bias.has_value() && bias->shape != normalized_shape) {
    std::ostringstream oss;
    oss << make_error_prefix() << "bias shape "
        << detail::shape_to_string(bias->shape)
        << " must match normalized_shape.";
    throw std::invalid_argument(oss.str());
  }

  const int64_t normalized_numel = detail::checked_numel(normalized_shape);
  if (normalized_numel <= 0) {
    throw std::invalid_argument(
        make_error_prefix() +
        "cannot normalize over zero elements in normalized_shape.");
  }

  const Tensor input_contiguous = input.contiguous();
  Tensor output(input.shape);

  std::optional<Tensor> weight_contiguous;
  std::optional<Tensor> bias_contiguous;
  const float *weight_ptr = nullptr;
  const float *bias_ptr = nullptr;

  if (weight.has_value()) {
    weight_contiguous = weight->contiguous();
    weight_ptr = weight_contiguous->data_ptr();
  }
  if (bias.has_value()) {
    bias_contiguous = bias->contiguous();
    bias_ptr = bias_contiguous->data_ptr();
  }

  const float *input_ptr = input_contiguous.data_ptr();
  float *output_ptr = output.data_ptr();
  const int64_t outer_numel = input.numel() / normalized_numel;

  for (int64_t outer_idx = 0; outer_idx < outer_numel; ++outer_idx) {
    const int64_t base = outer_idx * normalized_numel;

    float sum = 0.0f;
    for (int64_t i = 0; i < normalized_numel; ++i) {
      sum += input_ptr[base + i];
    }
    const float mean = sum / static_cast<float>(normalized_numel);

    float squared_sum = 0.0f;
    for (int64_t i = 0; i < normalized_numel; ++i) {
      const float centered = input_ptr[base + i] - mean;
      squared_sum += centered * centered;
    }
    const float variance = squared_sum / static_cast<float>(normalized_numel);
    const float inv_std = 1.0f / std::sqrt(variance + eps);

    for (int64_t i = 0; i < normalized_numel; ++i) {
      float value = (input_ptr[base + i] - mean) * inv_std;
      if (weight_ptr != nullptr) {
        value *= weight_ptr[i];
      }
      if (bias_ptr != nullptr) {
        value += bias_ptr[i];
      }
      output_ptr[base + i] = value;
    }
  }

  return output;
}

/*
 * Computes cross-entropy loss between logits and class-index targets.
 */
Tensor cross_entropy(const Tensor &input, const Tensor &target,
                     const int64_t ignore_index, const std::string &reduction) {
  validate_copy_metadata(input, "cross_entropy");
  validate_copy_metadata(target, "cross_entropy");
  if (should_record_binary(input, target)) {
    throw_autograd_not_implemented("cross_entropy");
  }

  const auto make_error_prefix = [&input, &target]() {
    return std::string("cross_entropy failed for input shape ") +
           detail::shape_to_string(input.shape) + " and target shape " +
           detail::shape_to_string(target.shape) + ": ";
  };

  if (input.ndim() < 1) {
    throw std::invalid_argument(make_error_prefix() +
                                "input must have rank >= 1 with shape [C] or "
                                "[N, C, ...].");
  }

  const int64_t class_dim = input.ndim() == 1 ? 0 : 1;
  const int64_t class_count = input.shape[static_cast<size_t>(class_dim)];
  if (class_count <= 0) {
    std::ostringstream oss;
    oss << make_error_prefix()
        << "input class dimension size must be positive, got " << class_count
        << ".";
    throw std::invalid_argument(oss.str());
  }

  std::vector<int64_t> expected_target_shape;
  expected_target_shape.reserve(input.shape.size() - 1);
  for (size_t input_dim = 0; input_dim < input.shape.size(); ++input_dim) {
    if (static_cast<int64_t>(input_dim) == class_dim) {
      continue;
    }
    expected_target_shape.push_back(input.shape[input_dim]);
  }
  if (target.shape != expected_target_shape) {
    std::ostringstream oss;
    oss << make_error_prefix() << "target shape must be "
        << detail::shape_to_string(expected_target_shape)
        << " to match input by removing the class dimension.";
    throw std::invalid_argument(oss.str());
  }

  enum class ReductionMode { kNone, kMean, kSum };
  ReductionMode reduction_mode;
  if (reduction == "none") {
    reduction_mode = ReductionMode::kNone;
  } else if (reduction == "mean") {
    reduction_mode = ReductionMode::kMean;
  } else if (reduction == "sum") {
    reduction_mode = ReductionMode::kSum;
  } else {
    throw std::invalid_argument(
        "cross_entropy() expected 'reduction' to be one of {'none', 'mean', "
        "'sum'}.");
  }

  const Tensor log_probs = input.log_softmax(class_dim);
  Tensor unreduced(target.shape);
  unreduced.storage->fill(0.0f);

  float total_loss = 0.0f;
  int64_t valid_count = 0;
  float *unreduced_ptr = unreduced.data_ptr();
  const float *target_ptr = target.data_ptr();
  const float *log_probs_ptr = log_probs.data_ptr();

  std::vector<int64_t> log_probs_target_strides(target.shape.size(), 0);
  size_t target_dim = 0;
  for (size_t input_dim = 0; input_dim < input.shape.size(); ++input_dim) {
    if (static_cast<int64_t>(input_dim) == class_dim) {
      continue;
    }
    log_probs_target_strides[target_dim] = log_probs.strides[input_dim];
    ++target_dim;
  }

  const int64_t target_numel = target.numel();
  std::vector<int64_t> coord(target.shape.size(), 0);
  int64_t target_offset = 0;
  int64_t unreduced_offset = 0;
  int64_t log_probs_base_offset = 0;

  for (int64_t linear_idx = 0; linear_idx < target_numel; ++linear_idx) {
    const float target_value = target_ptr[target_offset];
    if (!std::isfinite(target_value)) {
      throw std::invalid_argument(
          make_error_prefix() +
          "target values must be finite integer class indices.");
    }

    const float truncated = std::trunc(target_value);
    if (target_value != truncated) {
      std::ostringstream oss;
      oss << make_error_prefix()
          << "target values must be integer class indices.";
      throw std::invalid_argument(oss.str());
    }

    const int64_t class_index = static_cast<int64_t>(truncated);
    if (class_index == ignore_index) {
      unreduced_ptr[unreduced_offset] = 0.0f;
    } else {
      if (class_index < 0 || class_index >= class_count) {
        std::ostringstream oss;
        oss << make_error_prefix() << "target class index " << class_index
            << " is out of range for " << class_count << " classes.";
        throw std::invalid_argument(oss.str());
      }

      const float log_prob =
          log_probs_ptr[log_probs_base_offset +
                        class_index *
                            log_probs.strides[static_cast<size_t>(class_dim)]];
      const float loss = -log_prob;
      unreduced_ptr[unreduced_offset] = loss;
      total_loss += loss;
      ++valid_count;
    }

    if (target.shape.empty()) {
      continue;
    }

    for (int64_t dim = static_cast<int64_t>(target.shape.size()) - 1; dim >= 0;
         --dim) {
      const size_t dim_index = static_cast<size_t>(dim);
      ++coord[dim_index];
      target_offset += target.strides[dim_index];
      unreduced_offset += unreduced.strides[dim_index];
      log_probs_base_offset += log_probs_target_strides[dim_index];
      if (coord[dim_index] < target.shape[dim_index]) {
        break;
      }

      target_offset -= coord[dim_index] * target.strides[dim_index];
      unreduced_offset -= coord[dim_index] * unreduced.strides[dim_index];
      log_probs_base_offset -=
          coord[dim_index] * log_probs_target_strides[dim_index];
      coord[dim_index] = 0;
    }
  }

  if (reduction_mode == ReductionMode::kNone) {
    return unreduced;
  }

  Tensor reduced({});
  float *reduced_ptr = reduced.data_ptr();
  if (reduction_mode == ReductionMode::kSum) {
    *reduced_ptr = total_loss;
    return reduced;
  }

  if (valid_count == 0) {
    *reduced_ptr = std::numeric_limits<float>::quiet_NaN();
  } else {
    *reduced_ptr = total_loss / static_cast<float>(valid_count);
  }
  return reduced;
}

/*
 * Computes embedding lookup for integer index tensors.
 */
Tensor embedding(const Tensor &input, const Tensor &weight) {
  validate_copy_metadata(input, "embedding");
  validate_copy_metadata(weight, "embedding");
  if (should_record_binary(input, weight)) {
    throw_autograd_not_implemented("embedding");
  }

  const auto make_error_prefix = [&input, &weight]() {
    return std::string("embedding failed for input shape ") +
           detail::shape_to_string(input.shape) + " and weight shape " +
           detail::shape_to_string(weight.shape) + ": ";
  };

  if (weight.ndim() != 2) {
    throw std::invalid_argument(make_error_prefix() +
                                "weight must have rank 2 with shape [V, D].");
  }

  const int64_t vocab_size = weight.shape[0];
  const int64_t embedding_dim = weight.shape[1];
  if (vocab_size <= 0) {
    std::ostringstream oss;
    oss << make_error_prefix()
        << "weight.shape[0] (vocab size) must be positive, got " << vocab_size
        << ".";
    throw std::invalid_argument(oss.str());
  }

  std::vector<int64_t> output_shape = input.shape;
  output_shape.push_back(embedding_dim);
  Tensor output(output_shape);
  if (input.numel() == 0) {
    return output;
  }

  const float *input_ptr = input.data_ptr();
  const float *weight_ptr = weight.data_ptr();
  float *output_ptr = output.data_ptr();

  const int64_t input_rank = static_cast<int64_t>(input.shape.size());
  const int64_t input_numel = input.numel();
  std::vector<int64_t> coord(input.shape.size(), 0);
  int64_t input_offset = 0;
  int64_t output_offset = 0;

  for (int64_t linear_idx = 0; linear_idx < input_numel; ++linear_idx) {
    const float index_value = input_ptr[input_offset];
    if (!std::isfinite(index_value)) {
      throw std::invalid_argument(make_error_prefix() +
                                  "input indices must be finite integers.");
    }

    const float truncated = std::trunc(index_value);
    if (index_value != truncated) {
      throw std::invalid_argument(make_error_prefix() +
                                  "input indices must be integer-valued.");
    }

    const int64_t row_index = static_cast<int64_t>(truncated);
    if (row_index < 0 || row_index >= vocab_size) {
      std::ostringstream oss;
      oss << make_error_prefix() << "index " << row_index
          << " is out of range for vocab size " << vocab_size << ".";
      throw std::invalid_argument(oss.str());
    }

    const int64_t row_base_offset = row_index * weight.strides[0];
    for (int64_t d = 0; d < embedding_dim; ++d) {
      output_ptr[output_offset + (d * output.strides.back())] =
          weight_ptr[row_base_offset + (d * weight.strides[1])];
    }

    if (input_rank == 0) {
      continue;
    }

    for (int64_t dim = input_rank - 1; dim >= 0; --dim) {
      const size_t dim_index = static_cast<size_t>(dim);
      ++coord[dim_index];
      input_offset += input.strides[dim_index];
      output_offset += output.strides[dim_index];
      if (coord[dim_index] < input.shape[dim_index]) {
        break;
      }

      input_offset -= coord[dim_index] * input.strides[dim_index];
      output_offset -= coord[dim_index] * output.strides[dim_index];
      coord[dim_index] = 0;
    }
  }

  return output;
}

/*
 * Creates a tensor filled with a constant value.
 */
Tensor full(const std::vector<int64_t> &shape, const float fill_value,
            const bool requires_grad) {
  Tensor tensor(shape);
  tensor.storage->fill(fill_value);
  if (requires_grad) {
    tensor.requires_grad_(true);
  }
  return tensor;
}

/*
 * Creates a tensor filled with zeros.
 */
Tensor zeros(const std::vector<int64_t> &shape, const bool requires_grad) {
  return full(shape, 0.0f, requires_grad);
}

/*
 * Creates a tensor filled with ones.
 */
Tensor ones(const std::vector<int64_t> &shape, const bool requires_grad) {
  return full(shape, 1.0f, requires_grad);
}

} /* namespace bt */
