/*
 * File: native/src/tensor_reductions.cpp
 * Purpose: Implements tensor reduction ops and their autograd nodes.
 */

#include "bt/tensor.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "bt/detail/autograd_record.h"
#include "bt/detail/dims.h"
#include "bt/detail/format.h"
#include "bt/detail/tensor_validation.h"

/*
 * Namespace: (anonymous)
 * Purpose: Private implementation details local to this translation unit.
 */
namespace {

/*
 * Struct: ReductionPlan
 * Purpose: Stores precomputed metadata for a reduction.
 */
struct ReductionPlan {
  std::vector<bool> reduce_mask;
  std::vector<int64_t> input_to_output_dim;
  std::vector<int64_t> output_shape;
};

/*
 * Normalizes one reduction dimension using Python-style indexing.
 */
[[nodiscard]] int64_t normalize_reduction_dim(const bt::Tensor &tensor, const int64_t dim,
                                              const size_t dim_index,
                                              const std::string_view operation_name) {
  const int64_t rank = static_cast<int64_t>(tensor.shape.size());
  const int64_t normalized = dim < 0 ? dim + rank : dim;
  if (normalized >= 0 && normalized < rank) {
    return normalized;
  }

  std::ostringstream oss;
  oss << operation_name << " failed for tensor with shape "
      << bt::detail::shape_to_string(tensor.shape) << ": dim[" << dim_index << "]=" << dim
      << " is out of range for rank " << rank << ".";
  throw std::invalid_argument(oss.str());
}

/*
 * Normalizes and validates reduction dimensions.
 */
[[nodiscard]] std::vector<int64_t>
normalize_reduction_dims(const bt::Tensor &tensor, const std::vector<int64_t> &dim,
                         const std::string_view operation_name) {
  const int64_t rank = static_cast<int64_t>(tensor.shape.size());
  std::vector<int64_t> normalized_dims;
  normalized_dims.reserve(dim.size());

  std::vector<bool> seen(static_cast<size_t>(rank), false);
  for (size_t i = 0; i < dim.size(); ++i) {
    const int64_t normalized = normalize_reduction_dim(tensor, dim[i], i, operation_name);
    if (seen[static_cast<size_t>(normalized)]) {
      std::ostringstream oss;
      oss << operation_name << " failed for tensor with shape "
          << bt::detail::shape_to_string(tensor.shape) << ": dimension " << normalized
          << " appears more than once in dim.";
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
                     const std::vector<int64_t> &normalized_dims, const bool keepdim) {
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
                       plan.input_to_output_dim, tensor.data_ptr(), out.data_ptr());
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
                       plan.input_to_output_dim, tensor.data_ptr(), out.data_ptr());
  return out;
}

/*
 * Returns a reduction output gradient reshaped to keepdim form.
 */
[[nodiscard]] bt::Tensor expand_reduction_grad(const bt::Tensor &out_grad,
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

/*
 * Class: SumNode
 * Purpose: Backward pass for Tensor::sum.
 */
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
    const bt::Tensor grad = expand_reduction_grad(out_grad, reduced_dims_, keepdim_);
    const bt::Tensor expanded = grad * bt::ones(input_shape_);
    return {expanded};
  }

private:
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> reduced_dims_;
  bool keepdim_ = false;
};

/*
 * Class: MeanNode
 * Purpose: Backward pass for Tensor::mean.
 */
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
        expand_reduction_grad(out_grad, reduced_dims_, keepdim_) * bt::ones(input_shape_);
    return {expanded / static_cast<float>(reduced_count_)};
  }

private:
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> reduced_dims_;
  bool keepdim_ = false;
  int64_t reduced_count_ = 1;
};

/*
 * Recursively counts max ties for each reduced output position.
 */
void recursive_count_max_ties(const size_t dim, const std::vector<int64_t> &shape,
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

/*
 * Recursively scatters max backward gradients into input positions.
 */
void recursive_scatter_max_grad(const size_t dim, const std::vector<int64_t> &shape,
                                const std::vector<int64_t> &input_strides,
                                const std::vector<int64_t> &in_grad_strides,
                                const std::vector<int64_t> &output_strides,
                                const std::vector<int64_t> &input_to_output_dim,
                                const float *input_ptr, const float *max_ptr,
                                const float *out_grad_ptr, const float *count_ptr,
                                float *in_grad_ptr) {
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
      const int64_t output_stride = output_strides[static_cast<size_t>(out_dim)];
      next_max_ptr += index * output_stride;
      next_out_grad_ptr += index * output_stride;
      next_count_ptr += index * output_stride;
    }

    recursive_scatter_max_grad(dim + 1, shape, input_strides, in_grad_strides,
                               output_strides, input_to_output_dim, next_input_ptr,
                               next_max_ptr, next_out_grad_ptr, next_count_ptr,
                               next_in_grad_ptr);
  }
}

/*
 * Class: MaxNode
 * Purpose: Backward pass for Tensor::max reduction.
 */
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
    const bt::Tensor out_grad_keepdim_contiguous = out_grad_keepdim.contiguous();
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

    const ReductionPlan keepdim_plan = build_reduction_plan(input, reduced_dims_, true);
    recursive_count_max_ties(0, input.shape, input.strides,
                             max_keepdim_contiguous.strides,
                             keepdim_plan.input_to_output_dim, input.data_ptr(),
                             max_keepdim_contiguous.data_ptr(), tie_counts.data_ptr());
    recursive_scatter_max_grad(
        0, input.shape, input.strides, input_grad.strides, max_keepdim_contiguous.strides,
        keepdim_plan.input_to_output_dim, input.data_ptr(),
        max_keepdim_contiguous.data_ptr(), out_grad_keepdim_contiguous.data_ptr(),
        tie_counts.data_ptr(), input_grad.data_ptr());

    return {input_grad};
  }

private:
  std::vector<int64_t> reduced_dims_;
  bool keepdim_ = false;
};

} // namespace

/*
 * Namespace: bt
 * Purpose: Public BareTensor C++ API surface.
 */
namespace bt {

/*
 * Returns the sum of all tensor elements as a scalar tensor.
 */
Tensor Tensor::sum() const { return sum(detail::make_axis_order(shape.size()), false); }

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
  const ReductionPlan plan = build_reduction_plan(*this, normalized_dims, keepdim);
  Tensor out = sum_with_plan(*this, plan);
  if (bt::detail::should_record_unary(*this)) {
    out.set_grad_fn(std::make_shared<SumNode>(*this, normalized_dims, keepdim));
  }
  return out;
}

/*
 * Returns the mean of all tensor elements as a scalar tensor.
 */
Tensor Tensor::mean() const { return mean(detail::make_axis_order(shape.size()), false); }

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
  const ReductionPlan plan = build_reduction_plan(*this, normalized_dims, keepdim);
  const Tensor reduced_sum = sum_with_plan(*this, plan);
  const int64_t reduced_element_count = reduction_element_count(*this, normalized_dims);
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
Tensor Tensor::max() const { return max(detail::make_axis_order(shape.size()), false); }

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
  const ReductionPlan plan = build_reduction_plan(*this, normalized_dims, keepdim);
  Tensor out = max_with_plan(*this, plan);
  if (bt::detail::should_record_unary(*this)) {
    out.set_grad_fn(std::make_shared<MaxNode>(*this, normalized_dims, keepdim));
  }
  return out;
}

} /* namespace bt */
