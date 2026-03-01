/*
 * File: native/src/tensor_nn_ops.cpp
 * Purpose: Implements NN-oriented tensor forward ops.
 */

#include "bt/tensor.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bt/detail/autograd_record.h"
#include "bt/detail/dims.h"
#include "bt/detail/format.h"
#include "bt/detail/shape.h"
#include "bt/detail/tensor_nn_autograd.h"
#include "bt/detail/tensor_validation.h"

/*
 * Namespace: (anonymous)
 * Purpose: Private implementation details local to this translation unit.
 */
namespace {

/*
 * Parses cross-entropy reduction mode from a user-facing string.
 */
[[nodiscard]] bt::detail::CrossEntropyReductionMode
parse_cross_entropy_reduction(const std::string &reduction) {
  if (reduction == "none") {
    return bt::detail::CrossEntropyReductionMode::kNone;
  }
  if (reduction == "mean") {
    return bt::detail::CrossEntropyReductionMode::kMean;
  }
  if (reduction == "sum") {
    return bt::detail::CrossEntropyReductionMode::kSum;
  }

  throw std::invalid_argument(
      "cross_entropy() expected 'reduction' to be one of {'none', 'mean', "
      "'sum'}.");
}

} // namespace

/*
 * Namespace: bt
 * Purpose: Public BareTensor C++ API surface.
 */
namespace bt {

/*
 * Returns a tensor containing softmax values computed along dim.
 */
Tensor Tensor::softmax(const int64_t dim) const {
  bt::detail::validate_copy_metadata(*this, "softmax");
  const bool should_record = bt::detail::should_record_unary(*this);

  const int64_t normalized_dim =
      detail::normalize_dim_checked("softmax", shape, dim, "dim");
  if (shape[static_cast<size_t>(normalized_dim)] == 0) {
    Tensor out = exp();
    if (should_record) {
      out.set_grad_fn(bt::detail::make_softmax_node(*this, normalized_dim));
    }
    return out;
  }

  const Tensor max_values = max(normalized_dim, true);
  const Tensor shifted = (*this) - max_values;
  const Tensor exp_values = shifted.exp();
  const Tensor normalizer = exp_values.sum(normalized_dim, true);
  Tensor out = exp_values / normalizer;
  if (should_record) {
    out.set_grad_fn(bt::detail::make_softmax_node(*this, normalized_dim));
  }
  return out;
}

/*
 * Returns a tensor containing log-softmax values computed along dim.
 */
Tensor Tensor::log_softmax(const int64_t dim) const {
  bt::detail::validate_copy_metadata(*this, "log_softmax");
  const bool should_record = bt::detail::should_record_unary(*this);

  const int64_t normalized_dim =
      detail::normalize_dim_checked("log_softmax", shape, dim, "dim");
  if (shape[static_cast<size_t>(normalized_dim)] == 0) {
    Tensor out = log();
    if (should_record) {
      out.set_grad_fn(bt::detail::make_log_softmax_node(*this, normalized_dim));
    }
    return out;
  }

  const Tensor max_values = max(normalized_dim, true);
  const Tensor shifted = (*this) - max_values;
  const Tensor log_normalizer = shifted.exp().sum(normalized_dim, true).log();
  Tensor out = shifted - log_normalizer;
  if (should_record) {
    out.set_grad_fn(bt::detail::make_log_softmax_node(*this, normalized_dim));
  }
  return out;
}

/*
 * Applies layer normalization over the trailing normalized_shape dimensions.
 */
Tensor layer_norm(const Tensor &input,
                  const std::vector<int64_t> &normalized_shape,
                  const std::optional<Tensor> &weight,
                  const std::optional<Tensor> &bias, const float eps) {
  bt::detail::validate_copy_metadata(input, "layer_norm");
  const bool should_record =
      bt::detail::should_record_unary(input) ||
      (weight.has_value() && bt::detail::should_record_unary(*weight)) ||
      (bias.has_value() && bt::detail::should_record_unary(*bias));
  if (weight.has_value()) {
    bt::detail::validate_copy_metadata(*weight, "layer_norm");
  }
  if (bias.has_value()) {
    bt::detail::validate_copy_metadata(*bias, "layer_norm");
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

  if (should_record) {
    std::vector<Tensor> node_inputs{input};
    if (weight.has_value()) {
      node_inputs.push_back(*weight);
    }
    if (bias.has_value()) {
      node_inputs.push_back(*bias);
    }
    output.set_grad_fn(bt::detail::make_layer_norm_node(
        std::move(node_inputs), normalized_shape, eps, weight.has_value(),
        bias.has_value()));
  }

  return output;
}

/*
 * Computes cross-entropy loss between logits and class-index targets.
 */
Tensor cross_entropy(const Tensor &input, const Tensor &target,
                     const int64_t ignore_index, const std::string &reduction) {
  bt::detail::validate_copy_metadata(input, "cross_entropy");
  bt::detail::validate_copy_metadata(target, "cross_entropy");
  const bool should_record = bt::detail::should_record_binary(input, target);

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

  const bt::detail::CrossEntropyReductionMode reduction_mode =
      parse_cross_entropy_reduction(reduction);

  const Tensor log_probs = input.log_softmax(class_dim);
  Tensor unreduced = bt::full(target.shape, 0.0f);

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

  if (reduction_mode == bt::detail::CrossEntropyReductionMode::kNone) {
    if (should_record) {
      unreduced.set_grad_fn(bt::detail::make_cross_entropy_node(
          input, target, class_dim, ignore_index, reduction_mode));
    }
    return unreduced;
  }

  Tensor reduced({});
  float *reduced_ptr = reduced.data_ptr();
  if (reduction_mode == bt::detail::CrossEntropyReductionMode::kSum) {
    *reduced_ptr = total_loss;
    if (should_record) {
      reduced.set_grad_fn(bt::detail::make_cross_entropy_node(
          input, target, class_dim, ignore_index, reduction_mode));
    }
    return reduced;
  }

  if (valid_count == 0) {
    *reduced_ptr = std::numeric_limits<float>::quiet_NaN();
  } else {
    *reduced_ptr = total_loss / static_cast<float>(valid_count);
  }
  if (should_record) {
    reduced.set_grad_fn(bt::detail::make_cross_entropy_node(
        input, target, class_dim, ignore_index, reduction_mode));
  }
  return reduced;
}

/*
 * Computes embedding lookup for integer index tensors.
 */
Tensor embedding(const Tensor &input, const Tensor &weight) {
  bt::detail::validate_copy_metadata(input, "embedding");
  bt::detail::validate_copy_metadata(weight, "embedding");
  const bool should_record = bt::detail::should_record_binary(input, weight);

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
    if (should_record) {
      output.set_grad_fn(bt::detail::make_embedding_node(input, weight));
    }
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

  if (should_record) {
    output.set_grad_fn(bt::detail::make_embedding_node(input, weight));
  }
  return output;
}

} // namespace bt
