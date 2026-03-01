/*
 * File: native/src/tensor_nn_autograd.cpp
 * Purpose: Implements NN autograd node classes and node factories.
 */

#include "bt/detail/tensor_nn_autograd.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "bt/detail/shape.h"

/*
 * Namespace: (anonymous)
 * Purpose: Private implementation details local to this translation unit.
 */
namespace {

/*
 * Class: SoftmaxNode
 * Purpose: Backward formula for Tensor::softmax(dim).
 */
class SoftmaxNode final : public bt::Node {
public:
  SoftmaxNode(const bt::Tensor &input, const int64_t dim)
      : bt::Node({input}), dim_(dim) {}

  [[nodiscard]] std::vector<bt::Tensor>
  backward(const bt::Tensor &out_grad) const override {
    const bt::Tensor probs = this->inputs()[0].softmax(dim_);
    const bt::Tensor weighted_sum = (out_grad * probs).sum(dim_, true);
    return {probs * (out_grad - weighted_sum)};
  }

private:
  int64_t dim_ = 0;
};

/*
 * Class: LogSoftmaxNode
 * Purpose: Backward formula for Tensor::log_softmax(dim).
 */
class LogSoftmaxNode final : public bt::Node {
public:
  LogSoftmaxNode(const bt::Tensor &input, const int64_t dim)
      : bt::Node({input}), dim_(dim) {}

  [[nodiscard]] std::vector<bt::Tensor>
  backward(const bt::Tensor &out_grad) const override {
    const bt::Tensor probs = this->inputs()[0].softmax(dim_);
    const bt::Tensor out_grad_sum = out_grad.sum(dim_, true);
    return {out_grad - (probs * out_grad_sum)};
  }

private:
  int64_t dim_ = 0;
};

/*
 * Class: LayerNormNode
 * Purpose: Backward formula for layer normalization with optional affine
 * params.
 */
class LayerNormNode final : public bt::Node {
public:
  LayerNormNode(std::vector<bt::Tensor> inputs,
                std::vector<int64_t> normalized_shape, const float eps,
                const bool has_weight, const bool has_bias)
      : bt::Node(std::move(inputs)),
        normalized_shape_(std::move(normalized_shape)),
        normalized_numel_(bt::detail::checked_numel(normalized_shape_)),
        eps_(eps), has_weight_(has_weight), has_bias_(has_bias) {}

  [[nodiscard]] std::vector<bt::Tensor>
  backward(const bt::Tensor &out_grad) const override {
    const std::vector<bt::Tensor> &inputs = this->inputs();
    const bt::Tensor input_contiguous = inputs[0].contiguous();
    const bt::Tensor out_grad_contiguous = out_grad.contiguous();

    std::optional<bt::Tensor> weight_contiguous;
    const float *weight_ptr = nullptr;
    if (has_weight_) {
      weight_contiguous = inputs[1].contiguous();
      weight_ptr = weight_contiguous->data_ptr();
    }

    bt::Tensor input_grad = bt::full(inputs[0].shape, 0.0f);

    std::optional<bt::Tensor> weight_grad;
    float *weight_grad_ptr = nullptr;
    if (has_weight_) {
      weight_grad = bt::zeros(normalized_shape_);
      weight_grad_ptr = weight_grad->data_ptr();
    }

    std::optional<bt::Tensor> bias_grad;
    float *bias_grad_ptr = nullptr;
    if (has_bias_) {
      bias_grad = bt::zeros(normalized_shape_);
      bias_grad_ptr = bias_grad->data_ptr();
    }

    const float *input_ptr = input_contiguous.data_ptr();
    const float *out_grad_ptr = out_grad_contiguous.data_ptr();
    float *input_grad_ptr = input_grad.data_ptr();

    const int64_t outer_numel = input_contiguous.numel() / normalized_numel_;
    const float inv_normalized_numel =
        1.0f / static_cast<float>(normalized_numel_);
    for (int64_t outer_idx = 0; outer_idx < outer_numel; ++outer_idx) {
      const int64_t base = outer_idx * normalized_numel_;

      float sum = 0.0f;
      for (int64_t i = 0; i < normalized_numel_; ++i) {
        sum += input_ptr[base + i];
      }
      const float mean = sum * inv_normalized_numel;

      float sq_sum = 0.0f;
      for (int64_t i = 0; i < normalized_numel_; ++i) {
        const float centered = input_ptr[base + i] - mean;
        sq_sum += centered * centered;
      }
      const float variance = sq_sum * inv_normalized_numel;
      const float inv_std = 1.0f / std::sqrt(variance + eps_);

      float dx_hat_sum = 0.0f;
      float dx_hat_xhat_sum = 0.0f;
      for (int64_t i = 0; i < normalized_numel_; ++i) {
        const float x_hat = (input_ptr[base + i] - mean) * inv_std;
        const float dout = out_grad_ptr[base + i];
        const float dx_hat =
            weight_ptr != nullptr ? (dout * weight_ptr[i]) : dout;

        dx_hat_sum += dx_hat;
        dx_hat_xhat_sum += dx_hat * x_hat;

        if (weight_grad_ptr != nullptr) {
          weight_grad_ptr[i] += dout * x_hat;
        }
        if (bias_grad_ptr != nullptr) {
          bias_grad_ptr[i] += dout;
        }
      }

      for (int64_t i = 0; i < normalized_numel_; ++i) {
        const float x_hat = (input_ptr[base + i] - mean) * inv_std;
        const float dout = out_grad_ptr[base + i];
        const float dx_hat =
            weight_ptr != nullptr ? (dout * weight_ptr[i]) : dout;
        const float dx = inv_std * inv_normalized_numel *
                         (static_cast<float>(normalized_numel_) * dx_hat -
                          dx_hat_sum - (x_hat * dx_hat_xhat_sum));
        input_grad_ptr[base + i] = dx;
      }
    }

    std::vector<bt::Tensor> grads;
    grads.reserve(1 + static_cast<size_t>(has_weight_) +
                  static_cast<size_t>(has_bias_));
    grads.push_back(input_grad);
    if (has_weight_) {
      grads.push_back(*weight_grad);
    }
    if (has_bias_) {
      grads.push_back(*bias_grad);
    }
    return grads;
  }

private:
  std::vector<int64_t> normalized_shape_;
  int64_t normalized_numel_ = 1;
  float eps_ = 1e-5f;
  bool has_weight_ = false;
  bool has_bias_ = false;
};

/*
 * Class: CrossEntropyNode
 * Purpose: Backward formula for cross-entropy loss on class-index targets.
 */
class CrossEntropyNode final : public bt::Node {
public:
  CrossEntropyNode(const bt::Tensor &input, const bt::Tensor &target,
                   const int64_t class_dim, const int64_t ignore_index,
                   const bt::detail::CrossEntropyReductionMode reduction_mode)
      : bt::Node({input, target}), class_dim_(class_dim),
        ignore_index_(ignore_index), reduction_mode_(reduction_mode) {}

  [[nodiscard]] std::vector<bt::Tensor>
  backward(const bt::Tensor &out_grad) const override {
    const std::vector<bt::Tensor> &inputs = this->inputs();
    const bt::Tensor &input = inputs[0];
    const bt::Tensor target_contiguous = inputs[1].contiguous();
    const bt::Tensor probs_contiguous = input.softmax(class_dim_).contiguous();

    bt::Tensor input_grad = bt::full(input.shape, 0.0f);
    bt::Tensor target_grad = bt::zeros(inputs[1].shape);

    const int64_t class_count = input.shape[static_cast<size_t>(class_dim_)];
    const int64_t class_stride =
        probs_contiguous.strides[static_cast<size_t>(class_dim_)];
    const int64_t target_numel = target_contiguous.numel();

    const float *target_ptr = target_contiguous.data_ptr();
    const float *probs_ptr = probs_contiguous.data_ptr();
    float *input_grad_ptr = input_grad.data_ptr();

    int64_t valid_count = 0;
    if (reduction_mode_ == bt::detail::CrossEntropyReductionMode::kMean) {
      for (int64_t linear_idx = 0; linear_idx < target_numel; ++linear_idx) {
        const float target_value = target_ptr[linear_idx];
        if (!std::isfinite(target_value) ||
            target_value != std::trunc(target_value)) {
          throw std::runtime_error(
              "cross_entropy backward received invalid non-integer target "
              "value.");
        }

        const int64_t class_index = static_cast<int64_t>(target_value);
        if (class_index == ignore_index_) {
          continue;
        }
        if (class_index < 0 || class_index >= class_count) {
          throw std::runtime_error(
              "cross_entropy backward received out-of-range target class "
              "index.");
        }
        ++valid_count;
      }
    }

    const bt::Tensor out_grad_contiguous = out_grad.contiguous();
    const float *out_grad_ptr = out_grad_contiguous.data_ptr();

    float reduced_grad_scale = 0.0f;
    if (reduction_mode_ == bt::detail::CrossEntropyReductionMode::kSum) {
      reduced_grad_scale = out_grad_ptr[0];
    } else if (reduction_mode_ ==
               bt::detail::CrossEntropyReductionMode::kMean) {
      if (valid_count == 0) {
        return {input_grad, target_grad};
      }
      reduced_grad_scale = out_grad_ptr[0] / static_cast<float>(valid_count);
    }

    std::vector<int64_t> probs_target_strides(target_contiguous.shape.size(),
                                              0);
    size_t target_dim = 0;
    for (size_t input_dim = 0; input_dim < input.shape.size(); ++input_dim) {
      if (static_cast<int64_t>(input_dim) == class_dim_) {
        continue;
      }
      probs_target_strides[target_dim] = probs_contiguous.strides[input_dim];
      ++target_dim;
    }

    std::vector<int64_t> coord(target_contiguous.shape.size(), 0);
    int64_t probs_base_offset = 0;

    for (int64_t linear_idx = 0; linear_idx < target_numel; ++linear_idx) {
      const float target_value = target_ptr[linear_idx];
      if (!std::isfinite(target_value) ||
          target_value != std::trunc(target_value)) {
        throw std::runtime_error("cross_entropy backward received invalid "
                                 "non-integer target value.");
      }
      const int64_t class_index = static_cast<int64_t>(target_value);
      if (class_index != ignore_index_) {
        if (class_index < 0 || class_index >= class_count) {
          throw std::runtime_error(
              "cross_entropy backward received out-of-range target class "
              "index.");
        }

        const float grad_scale =
            reduction_mode_ == bt::detail::CrossEntropyReductionMode::kNone
                ? out_grad_ptr[linear_idx]
                : reduced_grad_scale;

        for (int64_t class_idx = 0; class_idx < class_count; ++class_idx) {
          const int64_t offset = probs_base_offset + (class_idx * class_stride);
          input_grad_ptr[offset] += grad_scale * probs_ptr[offset];
        }
        input_grad_ptr[probs_base_offset + (class_index * class_stride)] -=
            grad_scale;
      }

      if (target_contiguous.shape.empty()) {
        continue;
      }

      for (int64_t dim =
               static_cast<int64_t>(target_contiguous.shape.size()) - 1;
           dim >= 0; --dim) {
        const size_t dim_index = static_cast<size_t>(dim);
        ++coord[dim_index];
        probs_base_offset += probs_target_strides[dim_index];
        if (coord[dim_index] < target_contiguous.shape[dim_index]) {
          break;
        }

        probs_base_offset -= coord[dim_index] * probs_target_strides[dim_index];
        coord[dim_index] = 0;
      }
    }

    return {input_grad, target_grad};
  }

private:
  int64_t class_dim_ = 1;
  int64_t ignore_index_ = -100;
  bt::detail::CrossEntropyReductionMode reduction_mode_ =
      bt::detail::CrossEntropyReductionMode::kMean;
};

/*
 * Class: EmbeddingNode
 * Purpose: Backward scatter-add into embedding weights.
 */
class EmbeddingNode final : public bt::Node {
public:
  EmbeddingNode(const bt::Tensor &input, const bt::Tensor &weight)
      : bt::Node({input, weight}), embedding_dim_(weight.shape[1]) {}

  [[nodiscard]] std::vector<bt::Tensor>
  backward(const bt::Tensor &out_grad) const override {
    const std::vector<bt::Tensor> &inputs = this->inputs();
    const bt::Tensor input_contiguous = inputs[0].contiguous();
    const bt::Tensor out_grad_contiguous = out_grad.contiguous();

    bt::Tensor input_grad = bt::zeros(inputs[0].shape);
    bt::Tensor weight_grad = bt::zeros(inputs[1].shape);

    const int64_t vocab_size = inputs[1].shape[0];
    const int64_t input_numel = input_contiguous.numel();

    const float *input_ptr = input_contiguous.data_ptr();
    const float *out_grad_ptr = out_grad_contiguous.data_ptr();
    float *weight_grad_ptr = weight_grad.data_ptr();

    for (int64_t linear_idx = 0; linear_idx < input_numel; ++linear_idx) {
      const float index_value = input_ptr[linear_idx];
      if (!std::isfinite(index_value) ||
          index_value != std::trunc(index_value)) {
        throw std::runtime_error(
            "embedding backward received invalid non-integer index value.");
      }

      const int64_t row_index = static_cast<int64_t>(index_value);
      if (row_index < 0 || row_index >= vocab_size) {
        throw std::runtime_error(
            "embedding backward received out-of-range index value.");
      }

      const int64_t weight_row_offset = row_index * weight_grad.strides[0];
      const int64_t out_grad_offset = linear_idx * embedding_dim_;
      for (int64_t d = 0; d < embedding_dim_; ++d) {
        weight_grad_ptr[weight_row_offset + (d * weight_grad.strides[1])] +=
            out_grad_ptr[out_grad_offset +
                         (d * out_grad_contiguous.strides.back())];
      }
    }

    return {input_grad, weight_grad};
  }

private:
  int64_t embedding_dim_ = 0;
};

} // namespace

/*
 * Namespace: bt::detail
 * Purpose: Node factory functions used by NN forward ops.
 */
namespace bt::detail {

/*
 * Creates a backward node for Tensor::softmax(dim).
 */
[[nodiscard]] std::shared_ptr<Node> make_softmax_node(const Tensor &input,
                                                      const int64_t dim) {
  return std::make_shared<SoftmaxNode>(input, dim);
}

/*
 * Creates a backward node for Tensor::log_softmax(dim).
 */
[[nodiscard]] std::shared_ptr<Node> make_log_softmax_node(const Tensor &input,
                                                          const int64_t dim) {
  return std::make_shared<LogSoftmaxNode>(input, dim);
}

/*
 * Creates a backward node for layer_norm().
 */
[[nodiscard]] std::shared_ptr<Node>
make_layer_norm_node(std::vector<Tensor> inputs,
                     std::vector<int64_t> normalized_shape, const float eps,
                     const bool has_weight, const bool has_bias) {
  return std::make_shared<LayerNormNode>(std::move(inputs),
                                         std::move(normalized_shape), eps,
                                         has_weight, has_bias);
}

/*
 * Creates a backward node for cross_entropy().
 */
[[nodiscard]] std::shared_ptr<Node>
make_cross_entropy_node(const Tensor &input, const Tensor &target,
                        const int64_t class_dim, const int64_t ignore_index,
                        const CrossEntropyReductionMode reduction_mode) {
  return std::make_shared<CrossEntropyNode>(input, target, class_dim,
                                            ignore_index, reduction_mode);
}

/*
 * Creates a backward node for embedding().
 */
[[nodiscard]] std::shared_ptr<Node> make_embedding_node(const Tensor &input,
                                                        const Tensor &weight) {
  return std::make_shared<EmbeddingNode>(input, weight);
}

} // namespace bt::detail
