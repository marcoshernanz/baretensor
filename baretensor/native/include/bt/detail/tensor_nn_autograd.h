/*
 * File: native/include/bt/detail/tensor_nn_autograd.h
 * Purpose: Declares NN autograd node factory helpers shared by NN ops.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "bt/tensor.h"

/*
 * Namespace: bt::detail
 * Purpose: Internal reusable implementation helpers.
 */
namespace bt::detail {

/*
 * Encodes supported cross-entropy reduction modes.
 */
enum class CrossEntropyReductionMode { kNone, kMean, kSum };

/*
 * Creates a backward node for Tensor::softmax(dim).
 */
[[nodiscard]] std::shared_ptr<Node> make_softmax_node(const Tensor &input, int64_t dim);

/*
 * Creates a backward node for Tensor::log_softmax(dim).
 */
[[nodiscard]] std::shared_ptr<Node> make_log_softmax_node(const Tensor &input,
                                                          int64_t dim);

/*
 * Creates a backward node for layer_norm().
 */
[[nodiscard]] std::shared_ptr<Node>
make_layer_norm_node(std::vector<Tensor> inputs, std::vector<int64_t> normalized_shape,
                     float eps, bool has_weight, bool has_bias);

/*
 * Creates a backward node for cross_entropy().
 */
[[nodiscard]] std::shared_ptr<Node>
make_cross_entropy_node(const Tensor &input, const Tensor &target, int64_t class_dim,
                        int64_t ignore_index, CrossEntropyReductionMode reduction_mode);

/*
 * Creates a backward node for embedding().
 */
[[nodiscard]] std::shared_ptr<Node> make_embedding_node(const Tensor &input,
                                                        const Tensor &weight);

} // namespace bt::detail
