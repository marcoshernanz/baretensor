/*
 * File: native/include/bt/tensor.h
 * Purpose: Declares the tensor object and public tensor factory functions.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "bt/storage.h"

/*
 * Namespace: bt
 * Purpose: Public BareTensor C++ API surface.
 */
namespace bt {
class Node;
struct AutogradMeta;

/*
 * Class: Tensor
 * Purpose: Represents tensor metadata and storage-backed data access.
 */
class Tensor {
public:
  /*
   * Shared backing storage for tensor values.
   */
  std::shared_ptr<Storage> storage;

  /*
   * Element offset from the start of storage.
   */
  int64_t storage_offset = 0;

  /*
   * Tensor shape in element counts per dimension.
   */
  std::vector<int64_t> shape;

  /*
   * Tensor strides in elements per dimension.
   */
  std::vector<int64_t> strides;

  /*
   * Shared autograd metadata for this tensor handle.
   */
  std::shared_ptr<AutogradMeta> autograd_meta;

  /*
   * Constructs a tensor with allocated storage for the given shape.
   */
  Tensor(const std::vector<int64_t> &shape);

  /*
   * Constructs a tensor from owned data with the given shape.
   */
  Tensor(const std::vector<int64_t> &shape, std::vector<float> data);

  /*
   * Constructs a tensor view over existing storage and explicit metadata.
   * This constructor is intended for internal view-producing operations.
   */
  Tensor(const std::shared_ptr<Storage> storage, const int64_t storage_offset,
         const std::vector<int64_t> &shape,
         const std::vector<int64_t> &strides);

  /*
   * Returns the tensor rank.
   */
  [[nodiscard]] int ndim() const noexcept;

  /*
   * Returns the number of elements.
   */
  [[nodiscard]] int64_t numel() const noexcept;

  /*
   * Returns whether the tensor layout is contiguous.
   */
  [[nodiscard]] bool is_contiguous() const noexcept;

  /*
   * Returns a const data pointer at the tensor offset.
   */
  [[nodiscard]] const float *data_ptr() const noexcept;

  /*
   * Returns a mutable data pointer at the tensor offset.
   */
  [[nodiscard]] float *data_ptr() noexcept;

  /*
   * Returns whether autograd is enabled for this tensor.
   */
  [[nodiscard]] bool requires_grad() const noexcept;

  /*
   * Sets whether this tensor tracks gradients in autograd.
   */
  Tensor &set_requires_grad(bool requires_grad);

  /*
   * Returns whether this tensor is a leaf in the autograd graph.
   */
  [[nodiscard]] bool is_leaf() const noexcept;

  /*
   * Returns the accumulated gradient for this tensor, if available.
   */
  [[nodiscard]] std::optional<Tensor> grad() const;

  /*
   * Clears the accumulated gradient buffer for this tensor.
   */
  void zero_grad();

  /*
   * Returns a tensor sharing storage but detached from autograd history.
   */
  [[nodiscard]] Tensor detach() const;

  /*
   * Runs reverse-mode automatic differentiation from this tensor.
   */
  void backward(const std::optional<Tensor> &gradient = std::nullopt) const;

  /*
   * Assigns a gradient function node to this tensor.
   */
  void set_grad_fn(const std::shared_ptr<Node> &grad_fn);

  /*
   * Returns this tensor's gradient function node.
   */
  [[nodiscard]] std::shared_ptr<Node> grad_fn() const;

  /*
   * Accumulates a gradient tensor into this tensor's grad buffer.
   */
  void accumulate_grad(const Tensor &incoming_grad);

  /*
   * Returns a contiguous tensor with identical logical values and shape.
   * If the tensor is already contiguous, this returns an equivalent tensor
   * referencing the same storage.
   */
  [[nodiscard]] Tensor contiguous() const;

  /*
   * Returns a view of this tensor with the requested shape when the current
   * shape and strides are layout-compatible with the target view.
   * Supports a single inferred '-1' dimension in the requested shape.
   */
  [[nodiscard]] Tensor view(const std::vector<int64_t> &shape) const;

  /*
   * Returns a tensor with the requested shape, returning a view when possible
   * and otherwise returning a contiguous copy with the target shape.
   * Supports a single inferred '-1' dimension in the requested shape.
   */
  [[nodiscard]] Tensor reshape(const std::vector<int64_t> &shape) const;

  /*
   * Returns a view with dimensions reordered according to dims.
   * Supports negative dimensions using Python-style indexing and requires
   * dims to be a full permutation of [0, ..., ndim()-1].
   */
  [[nodiscard]] Tensor permute(const std::vector<int64_t> &dims) const;

  /*
   * Returns a view that selects one index along dim, removing that dimension.
   * Supports negative dimensions and negative indices using Python-style
   * indexing.
   */
  [[nodiscard]] Tensor select(int64_t dim, int64_t index) const;

  /*
   * Returns a strided view sliced along dim over [start, stop) with step.
   * Supports negative dimensions and Python-style slice bound normalization.
   * Requires step > 0.
   */
  [[nodiscard]] Tensor slice(int64_t dim, int64_t start, int64_t stop,
                             int64_t step = 1) const;

  /*
   * Returns a view with dim0 and dim1 swapped.
   * Supports negative dimensions using Python-style indexing.
   */
  [[nodiscard]] Tensor transpose(int64_t dim0, int64_t dim1) const;

  /*
   * Returns a 2-D matrix transpose view.
   * This operation requires ndim() == 2.
   */
  [[nodiscard]] Tensor T() const;

  /*
   * Returns a view with the last two dimensions swapped.
   * Equivalent to transpose(-2, -1).
   */
  [[nodiscard]] Tensor mT() const;

  /*
   * Returns the matrix product of this tensor and tensor2 using PyTorch-style
   * matmul semantics:
   * - 1D x 1D -> scalar dot product
   * - 2D x 2D -> matrix-matrix product
   * - 1D x 2D or 2D x 1D -> vector/matrix variants with singleton expansion
   * - N-D cases -> batched matrix multiplication with broadcasted batch dims
   * Both inputs must be at least 1-D.
   */
  [[nodiscard]] Tensor matmul(const Tensor &tensor2) const;

  /*
   * Returns the sum of all tensor elements as a scalar tensor.
   */
  [[nodiscard]] Tensor sum() const;

  /*
   * Returns the sum reduced along one dimension.
   * Supports negative dimensions using Python-style indexing.
   * If keepdim is true, the reduced dimension is retained with size 1.
   */
  [[nodiscard]] Tensor sum(int64_t dim, bool keepdim = false) const;

  /*
   * Returns the sum reduced along multiple dimensions.
   * Supports negative dimensions using Python-style indexing.
   * If keepdim is true, reduced dimensions are retained with size 1.
   * An empty dim list performs no reduction and returns a copy-equivalent
   * tensor with the same shape and values.
   */
  [[nodiscard]] Tensor sum(const std::vector<int64_t> &dim,
                           bool keepdim = false) const;

  /*
   * Returns the mean of all tensor elements as a scalar tensor.
   */
  [[nodiscard]] Tensor mean() const;

  /*
   * Returns the mean reduced along one dimension.
   * Supports negative dimensions using Python-style indexing.
   * If keepdim is true, the reduced dimension is retained with size 1.
   */
  [[nodiscard]] Tensor mean(int64_t dim, bool keepdim = false) const;

  /*
   * Returns the mean reduced along multiple dimensions.
   * Supports negative dimensions using Python-style indexing.
   * If keepdim is true, reduced dimensions are retained with size 1.
   * An empty dim list performs no reduction and returns a copy-equivalent
   * tensor with the same shape and values.
   */
  [[nodiscard]] Tensor mean(const std::vector<int64_t> &dim,
                            bool keepdim = false) const;

  /*
   * Returns the maximum of all tensor elements as a scalar tensor.
   */
  [[nodiscard]] Tensor max() const;

  /*
   * Returns the maximum reduced along one dimension.
   * Supports negative dimensions using Python-style indexing.
   * If keepdim is true, the reduced dimension is retained with size 1.
   */
  [[nodiscard]] Tensor max(int64_t dim, bool keepdim = false) const;

  /*
   * Returns the maximum reduced along multiple dimensions.
   * Supports negative dimensions using Python-style indexing.
   * If keepdim is true, reduced dimensions are retained with size 1.
   * An empty dim list performs no reduction and returns a copy-equivalent
   * tensor with the same shape and values.
   */
  [[nodiscard]] Tensor max(const std::vector<int64_t> &dim,
                           bool keepdim = false) const;

  /*
   * Returns a tensor containing the elementwise exponential of this tensor.
   * For each element x, the output contains exp(x).
   */
  [[nodiscard]] Tensor exp() const;

  /*
   * Returns a tensor containing the elementwise natural logarithm of this
   * tensor. For each element x, the output contains log(x).
   */
  [[nodiscard]] Tensor log() const;

  /*
   * Returns a tensor containing softmax values computed along dim.
   * Uses numerically stable normalization by subtracting the per-slice maximum
   * before exponentiation.
   * Supports negative dimensions using Python-style indexing.
   */
  [[nodiscard]] Tensor softmax(int64_t dim) const;

  /*
   * Returns a tensor containing log-softmax values computed along dim.
   * Uses numerically stable log-sum-exp normalization.
   * Supports negative dimensions using Python-style indexing.
   */
  [[nodiscard]] Tensor log_softmax(int64_t dim) const;

  /*
   * Elementwise tensor-tensor addition.
   */
  Tensor operator+(const Tensor &t) const;

  /*
   * Elementwise tensor-scalar addition.
   */
  Tensor operator+(float rhs) const;

  /*
   * Elementwise tensor-tensor subtraction.
   */
  Tensor operator-(const Tensor &t) const;

  /*
   * Elementwise tensor-scalar subtraction.
   */
  Tensor operator-(float rhs) const;

  /*
   * Elementwise tensor-tensor multiplication.
   */
  Tensor operator*(const Tensor &t) const;

  /*
   * Elementwise tensor-scalar multiplication.
   */
  Tensor operator*(float rhs) const;

  /*
   * Elementwise tensor-tensor division.
   */
  Tensor operator/(const Tensor &t) const;

  /*
   * Elementwise tensor-scalar division.
   */
  Tensor operator/(float rhs) const;
};

/*
 * Class: Node
 * Purpose: Represents a backward function in the autograd graph.
 */
class Node {
public:
  explicit Node(std::vector<Tensor> inputs);
  virtual ~Node() = default;

  /*
   * Returns the input tensors connected to this node.
   */
  [[nodiscard]] const std::vector<Tensor> &inputs() const noexcept;

  /*
   * Computes gradients for each input tensor from an output gradient.
   */
  [[nodiscard]] virtual std::vector<Tensor>
  backward(const Tensor &out_grad) const = 0;

private:
  std::vector<Tensor> inputs_;
};

/*
 * Struct: AutogradMeta
 * Purpose: Stores gradient tracking data shared across tensor aliases.
 */
struct AutogradMeta {
  bool requires_grad = false;
  bool is_leaf = true;
  std::optional<Tensor> grad = std::nullopt;
  std::shared_ptr<Node> grad_fn = nullptr;
};

/*
 * Namespace: autograd
 * Purpose: Runtime utilities for dynamic-graph differentiation.
 */
namespace autograd {

/*
 * Returns whether gradient recording is currently enabled.
 */
[[nodiscard]] bool is_grad_enabled() noexcept;

/*
 * RAII guard that disables gradient recording within a scope.
 */
class NoGradGuard {
public:
  NoGradGuard();
  ~NoGradGuard();

  NoGradGuard(const NoGradGuard &) = delete;
  NoGradGuard &operator=(const NoGradGuard &) = delete;

private:
  bool previous_state_ = true;
};

/*
 * Sums broadcasted gradient dimensions to match a target input shape.
 */
[[nodiscard]] Tensor reduce_sum_to_shape(const Tensor &grad,
                                         const std::vector<int64_t> &shape);

/*
 * Executes reverse-mode automatic differentiation from the output tensor.
 */
void backward(const Tensor &output,
              const std::optional<Tensor> &gradient = std::nullopt);

} // namespace autograd

/*
 * Creates a tensor filled with a constant value.
 */
[[nodiscard]] Tensor full(const std::vector<int64_t> &shape, float fill_value,
                          bool requires_grad = false);

/*
 * Creates a tensor filled with zeros.
 */
[[nodiscard]] Tensor zeros(const std::vector<int64_t> &shape,
                           bool requires_grad = false);

/*
 * Creates a tensor filled with ones.
 */
[[nodiscard]] Tensor ones(const std::vector<int64_t> &shape,
                          bool requires_grad = false);

/*
 * Computes cross-entropy loss between logits and class-index targets.
 * TinyGPT-focused scope:
 * - input shape [C] is supported with scalar target []
 * - input shape must be [N, C, d1, ..., dK] with K >= 0
 * - target shape must be [N, d1, ..., dK]
 * - target values must be integer class indices in [0, C) or ignore_index
 * - reduction supports "none", "mean", and "sum"
 */
[[nodiscard]] Tensor cross_entropy(const Tensor &input, const Tensor &target,
                                   int64_t ignore_index = -100,
                                   const std::string &reduction = "mean");

/*
 * Applies layer normalization over the trailing normalized_shape dimensions.
 * - normalized_shape must match the input tail dimensions exactly
 * - weight and bias, when provided, must have shape normalized_shape
 * - eps must be finite and greater than zero
 */
[[nodiscard]] Tensor
layer_norm(const Tensor &input, const std::vector<int64_t> &normalized_shape,
           const std::optional<Tensor> &weight = std::nullopt,
           const std::optional<Tensor> &bias = std::nullopt, float eps = 1e-5f);

/*
 * Computes embedding lookup for integer index tensors.
 * - input shape can be arbitrary and stores class indices
 * - weight shape must be [V, embedding_dim]
 * - output shape is input.shape + [embedding_dim]
 */
[[nodiscard]] Tensor embedding(const Tensor &input, const Tensor &weight);

} /* namespace bt */
