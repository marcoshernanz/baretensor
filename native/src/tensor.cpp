/*
 * File: native/src/tensor.cpp
 * Purpose: Implements tensor construction, metadata queries, and factories.
 */

#include "bt/tensor.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "bt/detail/autograd_record.h"
#include "bt/detail/broadcast.h"
#include "bt/detail/dims.h"
#include "bt/detail/format.h"
#include "bt/detail/shape.h"
#include "bt/detail/tensor_validation.h"

/*
 * Namespace: (anonymous)
 * Purpose: Private implementation details local to this translation unit.
 */
namespace {

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

[[nodiscard]] bt::Tensor
expand_reduction_grad(const bt::Tensor &out_grad,
                      const std::vector<int64_t> &reduced_dims, bool keepdim);

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
    const bt::Tensor grad =
        expand_reduction_grad(out_grad, reduced_dims_, keepdim_);
    const bt::Tensor expanded = grad * bt::ones(input_shape_);
    return {expanded};
  }

private:
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> reduced_dims_;
  bool keepdim_ = false;
};

[[nodiscard]] bt::Tensor
expand_reduction_grad(const bt::Tensor &out_grad,
                      const std::vector<int64_t> &reduced_dims,
                      const bool keepdim) {
  if (keepdim) {
    return out_grad;
  }

  std::vector<int64_t> reshape_shape = out_grad.shape;
  for (const int64_t dim : reduced_dims) {
    reshape_shape.insert(reshape_shape.begin() + dim, 1);
  }
  return out_grad.reshape(reshape_shape);
}

class MeanNode final : public bt::Node {
public:
  MeanNode(const bt::Tensor &input, const std::vector<int64_t> &reduced_dims,
           const bool keepdim, const int64_t reduced_count)
      : bt::Node({input}), input_shape_(input.shape), keepdim_(keepdim),
        reduced_count_(reduced_count) {
    reduced_dims_ = reduced_dims;
    std::sort(reduced_dims_.begin(), reduced_dims_.end());
  }

  [[nodiscard]] std::vector<bt::Tensor>
  backward(const bt::Tensor &out_grad) const override {
    const bt::Tensor expanded =
        expand_reduction_grad(out_grad, reduced_dims_, keepdim_) *
        bt::ones(input_shape_);
    return {expanded / static_cast<float>(reduced_count_)};
  }

private:
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> reduced_dims_;
  bool keepdim_ = false;
  int64_t reduced_count_ = 1;
};

void recursive_count_max_ties(const size_t dim,
                              const std::vector<int64_t> &shape,
                              const std::vector<int64_t> &input_strides,
                              const std::vector<int64_t> &output_strides,
                              const std::vector<int64_t> &input_to_output_dim,
                              const float *input_ptr, const float *max_ptr,
                              float *count_ptr) {
  if (dim == shape.size()) {
    if (*input_ptr == *max_ptr) {
      *count_ptr += 1.0f;
    }
    return;
  }

  if (shape[dim] == 0) {
    return;
  }

  const int64_t out_dim = input_to_output_dim[dim];
  for (int64_t index = 0; index < shape[dim]; ++index) {
    const float *next_input_ptr = input_ptr + (index * input_strides[dim]);
    const float *next_max_ptr = max_ptr;
    float *next_count_ptr = count_ptr;
    if (out_dim >= 0) {
      next_max_ptr += index * output_strides[static_cast<size_t>(out_dim)];
      next_count_ptr += index * output_strides[static_cast<size_t>(out_dim)];
    }

    recursive_count_max_ties(dim + 1, shape, input_strides, output_strides,
                             input_to_output_dim, next_input_ptr, next_max_ptr,
                             next_count_ptr);
  }
}

void recursive_scatter_max_grad(const size_t dim,
                                const std::vector<int64_t> &shape,
                                const std::vector<int64_t> &input_strides,
                                const std::vector<int64_t> &in_grad_strides,
                                const std::vector<int64_t> &output_strides,
                                const std::vector<int64_t> &input_to_output_dim,
                                const float *input_ptr, const float *max_ptr,
                                const float *out_grad_ptr,
                                const float *count_ptr, float *in_grad_ptr) {
  if (dim == shape.size()) {
    if (*input_ptr == *max_ptr && *count_ptr > 0.0f) {
      *in_grad_ptr = *out_grad_ptr / *count_ptr;
      return;
    }
    *in_grad_ptr = 0.0f;
    return;
  }

  if (shape[dim] == 0) {
    return;
  }

  const int64_t out_dim = input_to_output_dim[dim];
  for (int64_t index = 0; index < shape[dim]; ++index) {
    const float *next_input_ptr = input_ptr + (index * input_strides[dim]);
    const float *next_max_ptr = max_ptr;
    const float *next_out_grad_ptr = out_grad_ptr;
    const float *next_count_ptr = count_ptr;
    float *next_in_grad_ptr = in_grad_ptr + (index * in_grad_strides[dim]);
    if (out_dim >= 0) {
      const int64_t output_stride =
          output_strides[static_cast<size_t>(out_dim)];
      next_max_ptr += index * output_stride;
      next_out_grad_ptr += index * output_stride;
      next_count_ptr += index * output_stride;
    }

    recursive_scatter_max_grad(dim + 1, shape, input_strides, in_grad_strides,
                               output_strides, input_to_output_dim,
                               next_input_ptr, next_max_ptr, next_out_grad_ptr,
                               next_count_ptr, next_in_grad_ptr);
  }
}

class MaxNode final : public bt::Node {
public:
  MaxNode(const bt::Tensor &input, const std::vector<int64_t> &reduced_dims,
          const bool keepdim)
      : bt::Node({input}), keepdim_(keepdim) {
    reduced_dims_ = reduced_dims;
    std::sort(reduced_dims_.begin(), reduced_dims_.end());
  }

  [[nodiscard]] std::vector<bt::Tensor>
  backward(const bt::Tensor &out_grad) const override {
    const bt::Tensor &input = this->inputs()[0];
    const bt::Tensor out_grad_keepdim =
        expand_reduction_grad(out_grad, reduced_dims_, keepdim_);
    const bt::Tensor out_grad_keepdim_contiguous =
        out_grad_keepdim.contiguous();
    const bt::Tensor max_keepdim = input.max(reduced_dims_, true);
    const bt::Tensor max_keepdim_contiguous = max_keepdim.contiguous();

    bt::Tensor tie_counts = bt::full(max_keepdim_contiguous.shape, 0.0f);
    bt::Tensor input_grad = bt::full(input.shape, 0.0f);

    if (input.shape.empty()) {
      const float input_value = *input.data_ptr();
      const float max_value = *max_keepdim_contiguous.data_ptr();
      if (input_value == max_value) {
        *tie_counts.data_ptr() = 1.0f;
        *input_grad.data_ptr() = *out_grad_keepdim_contiguous.data_ptr();
      }
      return {input_grad};
    }

    const ReductionPlan keepdim_plan =
        build_reduction_plan(input, reduced_dims_, true);
    recursive_count_max_ties(
        0, input.shape, input.strides, max_keepdim_contiguous.strides,
        keepdim_plan.input_to_output_dim, input.data_ptr(),
        max_keepdim_contiguous.data_ptr(), tie_counts.data_ptr());
    recursive_scatter_max_grad(
        0, input.shape, input.strides, input_grad.strides,
        max_keepdim_contiguous.strides, keepdim_plan.input_to_output_dim,
        input.data_ptr(), max_keepdim_contiguous.data_ptr(),
        out_grad_keepdim_contiguous.data_ptr(), tie_counts.data_ptr(),
        input_grad.data_ptr());

    return {input_grad};
  }

private:
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
  bt::detail::validate_copy_metadata(*this, "contiguous");

  if (is_contiguous()) {
    return *this;
  }

  Tensor out(shape);
  bt::detail::validate_copy_metadata(out, "contiguous");

  const size_t ndim = shape.size();
  if (ndim == 0) {
    *out.data_ptr() = *data_ptr();
    return out;
  }

  recursive_copy(0, ndim, shape, data_ptr(), out.data_ptr(), strides,
                 out.strides);

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

  std::vector<int64_t> target_shape =
      detail::infer_reshape_shape(this->shape, shape);
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

  Tensor out(storage, storage_offset, std::move(target_shape),
             std::move(target_strides));
  if (bt::detail::should_record_unary(*this)) {
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
  bt::detail::validate_copy_metadata(*this, "transpose");

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

  std::vector<int64_t> dims = make_axis_order(shape.size());
  std::swap(dims[dims.size() - 2], dims[dims.size() - 1]);
  return permute(dims);
}

/*
 * Returns the matrix product of this tensor and tensor2 using matmul
 * semantics equivalent to PyTorch for dense tensors.
 */
Tensor Tensor::matmul(const Tensor &tensor2) const {
  bt::detail::validate_copy_metadata(*this, "matmul");
  bt::detail::validate_copy_metadata(tensor2, "matmul");

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
  if (bt::detail::should_record_binary(*this, tensor2)) {
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
  bt::detail::validate_copy_metadata(*this, "sum");

  const std::vector<int64_t> normalized_dims =
      normalize_reduction_dims(*this, dim, "sum");
  const ReductionPlan plan =
      build_reduction_plan(*this, normalized_dims, keepdim);
  Tensor out = sum_with_plan(*this, plan);
  if (bt::detail::should_record_unary(*this)) {
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
  bt::detail::validate_copy_metadata(*this, "mean");

  const std::vector<int64_t> normalized_dims =
      normalize_reduction_dims(*this, dim, "mean");
  const ReductionPlan plan =
      build_reduction_plan(*this, normalized_dims, keepdim);
  const Tensor reduced_sum = sum_with_plan(*this, plan);
  const int64_t reduced_element_count =
      reduction_element_count(*this, normalized_dims);
  Tensor out = reduced_sum / static_cast<float>(reduced_element_count);
  if (bt::detail::should_record_unary(*this)) {
    out.set_grad_fn(std::make_shared<MeanNode>(*this, normalized_dims, keepdim,
                                               reduced_element_count));
  }
  return out;
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
  bt::detail::validate_copy_metadata(*this, "max");

  const std::vector<int64_t> normalized_dims =
      normalize_reduction_dims(*this, dim, "max");
  validate_non_empty_reduction(*this, normalized_dims, "max");
  const ReductionPlan plan =
      build_reduction_plan(*this, normalized_dims, keepdim);
  Tensor out = max_with_plan(*this, plan);
  if (bt::detail::should_record_unary(*this)) {
    out.set_grad_fn(std::make_shared<MaxNode>(*this, normalized_dims, keepdim));
  }
  return out;
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
