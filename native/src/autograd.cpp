/*
 * File: native/src/autograd.cpp
 * Purpose: Implements dynamic-graph autograd runtime utilities.
 */

#include "bt/tensor.h"

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "bt/detail/format.h"

namespace {
thread_local bool g_grad_enabled = true;

void build_topology_from_tensor(const bt::Tensor &tensor,
                                std::unordered_set<const bt::Node *> &visited,
                                std::vector<std::shared_ptr<bt::Node>> &topo) {
  if (!tensor.requires_grad()) {
    return;
  }

  const std::shared_ptr<bt::Node> fn = tensor.grad_fn();
  if (fn == nullptr) {
    return;
  }

  const bt::Node *key = fn.get();
  if (visited.contains(key)) {
    return;
  }
  visited.insert(key);

  for (const bt::Tensor &input : fn->inputs()) {
    build_topology_from_tensor(input, visited, topo);
  }

  topo.push_back(fn);
}

[[nodiscard]] bt::Tensor make_default_root_grad(const bt::Tensor &output) {
  if (output.numel() != 1) {
    std::ostringstream oss;
    oss << "backward() requires an explicit gradient for non-scalar outputs "
           "(numel() must be 1), but got shape "
        << bt::detail::shape_to_string(output.shape) << ".";
    throw std::invalid_argument(oss.str());
  }
  return bt::ones(output.shape);
}

void validate_root_gradient_shape(const bt::Tensor &output, const bt::Tensor &gradient) {
  if (gradient.shape == output.shape) {
    return;
  }

  std::ostringstream oss;
  oss << "backward() gradient shape mismatch: output shape "
      << bt::detail::shape_to_string(output.shape) << " but got gradient shape "
      << bt::detail::shape_to_string(gradient.shape) << ".";
  throw std::invalid_argument(oss.str());
}

void validate_input_gradient_shape(const bt::Tensor &input,
                                   const bt::Tensor &input_grad) {
  if (input_grad.shape == input.shape) {
    return;
  }

  std::ostringstream oss;
  oss << "autograd node produced gradient shape "
      << bt::detail::shape_to_string(input_grad.shape) << " for input tensor shape "
      << bt::detail::shape_to_string(input.shape) << ".";
  throw std::runtime_error(oss.str());
}

} // namespace

namespace bt {

Node::Node(std::vector<Tensor> inputs) : inputs_(std::move(inputs)) {}

const std::vector<Tensor> &Node::inputs() const noexcept { return inputs_; }

namespace autograd {

bool is_grad_enabled() noexcept { return g_grad_enabled; }

NoGradGuard::NoGradGuard() : previous_state_(g_grad_enabled) { g_grad_enabled = false; }

NoGradGuard::~NoGradGuard() { g_grad_enabled = previous_state_; }

Tensor reduce_sum_to_shape(const Tensor &grad, const std::vector<int64_t> &shape) {
  if (grad.shape == shape) {
    return grad;
  }

  if (shape.size() > grad.shape.size()) {
    std::ostringstream oss;
    oss << "reduce_sum_to_shape failed: cannot reduce gradient shape "
        << detail::shape_to_string(grad.shape) << " to higher-rank shape "
        << detail::shape_to_string(shape) << ".";
    throw std::invalid_argument(oss.str());
  }

  Tensor reduced = grad;

  const size_t rank_diff = reduced.shape.size() - shape.size();
  if (rank_diff > 0) {
    std::vector<int64_t> leading_dims(rank_diff, 0);
    for (size_t i = 0; i < rank_diff; ++i) {
      leading_dims[i] = static_cast<int64_t>(i);
    }
    reduced = reduced.sum(leading_dims, false);
  }

  for (size_t dim = 0; dim < shape.size(); ++dim) {
    const int64_t current = reduced.shape[dim];
    const int64_t target = shape[dim];
    if (current == target) {
      continue;
    }

    if (target == 1 && current > 1) {
      reduced = reduced.sum(static_cast<int64_t>(dim), true);
      continue;
    }

    std::ostringstream oss;
    oss << "reduce_sum_to_shape failed: gradient shape "
        << detail::shape_to_string(grad.shape) << " cannot be reduced to "
        << detail::shape_to_string(shape) << ".";
    throw std::invalid_argument(oss.str());
  }

  if (reduced.shape != shape) {
    std::ostringstream oss;
    oss << "reduce_sum_to_shape failed: reduced gradient shape "
        << detail::shape_to_string(reduced.shape) << " does not match target shape "
        << detail::shape_to_string(shape) << ".";
    throw std::runtime_error(oss.str());
  }

  return reduced;
}

void backward(const Tensor &output, const std::optional<Tensor> &gradient) {
  if (!output.requires_grad()) {
    throw std::invalid_argument(
        "backward() called on a tensor that does not require gradients.");
  }

  NoGradGuard guard;

  Tensor root_grad =
      gradient.has_value() ? gradient.value() : make_default_root_grad(output);
  validate_root_gradient_shape(output, root_grad);

  const std::shared_ptr<Node> root_fn = output.grad_fn();
  if (root_fn == nullptr) {
    Tensor leaf = output;
    leaf.accumulate_grad(root_grad);
    return;
  }

  std::unordered_set<const Node *> visited;
  std::vector<std::shared_ptr<Node>> topo;
  build_topology_from_tensor(output, visited, topo);

  std::unordered_map<const Node *, Tensor> node_grads;
  node_grads.emplace(root_fn.get(), root_grad);

  for (size_t i = topo.size(); i > 0; --i) {
    const std::shared_ptr<Node> &node = topo[i - 1];
    auto grad_it = node_grads.find(node.get());
    if (grad_it == node_grads.end()) {
      continue;
    }

    const Tensor out_grad = grad_it->second;
    const std::vector<Tensor> input_grads = node->backward(out_grad);
    const std::vector<Tensor> &inputs = node->inputs();

    if (input_grads.size() != inputs.size()) {
      throw std::runtime_error(
          "autograd node returned an unexpected number of input gradients.");
    }

    for (size_t input_index = 0; input_index < inputs.size(); ++input_index) {
      const Tensor &input = inputs[input_index];
      if (!input.requires_grad()) {
        continue;
      }

      const Tensor input_grad = input_grads[input_index];
      validate_input_gradient_shape(input, input_grad);

      const std::shared_ptr<Node> input_fn = input.grad_fn();
      if (input_fn == nullptr) {
        Tensor leaf = input;
        leaf.accumulate_grad(input_grad);
        continue;
      }

      auto parent_grad_it = node_grads.find(input_fn.get());
      if (parent_grad_it == node_grads.end()) {
        node_grads.emplace(input_fn.get(), input_grad);
      } else {
        parent_grad_it->second = parent_grad_it->second + input_grad;
      }
    }

    node_grads.erase(grad_it);
  }
}

} // namespace autograd
} // namespace bt
