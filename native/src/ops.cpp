/*
 * File: native/src/ops.cpp
 * Purpose: Implements elementwise tensor operations and scalar variants.
 */

#include "bt/ops.h"

#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "bt/detail/broadcast.h"
#include "bt/tensor.h"

/*
 * Namespace: (anonymous)
 * Purpose: Private implementation details local to this translation unit.
 */
namespace {

/*
 * Applies a binary operation over two strided inputs by recursively traversing
 * N-D shape space.
 */
template <class Op>
void recursive_apply_binary(int dim, int ndim,
                            const std::vector<int64_t> &shape, const float *lhs,
                            const float *rhs, float *out,
                            const std::vector<int64_t> &lhs_strides,
                            const std::vector<int64_t> &rhs_strides,
                            const std::vector<int64_t> &out_strides,
                            const Op &op) {
  if (shape[dim] == 0)
    return;
  if (dim == ndim - 1) {
    for (int64_t i = 0; i < shape[dim]; ++i) {
      *out = op(*lhs, *rhs);
      lhs += lhs_strides[dim];
      rhs += rhs_strides[dim];
      out += out_strides[dim];
    }
    return;
  }

  for (int64_t i = 0; i < shape[dim]; ++i) {
    recursive_apply_binary(dim + 1, ndim, shape, lhs, rhs, out, lhs_strides,
                           rhs_strides, out_strides, op);
    lhs += lhs_strides[dim];
    rhs += rhs_strides[dim];
    out += out_strides[dim];
  }
}

/*
 * Applies a unary operation over one strided input by recursively traversing
 * N-D shape space.
 */
template <class Op>
void recursive_apply_unary(int dim, int ndim, const std::vector<int64_t> &shape,
                           const float *input, float *out,
                           const std::vector<int64_t> &input_strides,
                           const std::vector<int64_t> &out_strides,
                           const Op &op) {
  if (shape[dim] == 0)
    return;
  if (dim == ndim - 1) {
    for (int64_t i = 0; i < shape[dim]; ++i) {
      *out = op(*input);
      input += input_strides[dim];
      out += out_strides[dim];
    }
    return;
  }

  for (int64_t i = 0; i < shape[dim]; ++i) {
    recursive_apply_unary(dim + 1, ndim, shape, input, out, input_strides,
                          out_strides, op);
    input += input_strides[dim];
    out += out_strides[dim];
  }
}

/*
 * Executes a tensor-tensor elementwise operation with broadcasting support.
 */
template <class Op>
bt::Tensor binary_tt(const bt::Tensor &a, const bt::Tensor &b, Op op) {
  const std::vector<int64_t> out_shape =
      bt::detail::infer_broadcast_shape(a.shape, b.shape);
  bt::Tensor out(out_shape);

  const int64_t n = out.numel();
  if (n == 0)
    return out;

  const bool no_broadcast = (a.shape == out_shape) && (b.shape == out_shape);
  if (no_broadcast && a.is_contiguous() && b.is_contiguous() &&
      out.is_contiguous()) {
    const float *a_ptr = a.data_ptr();
    const float *b_ptr = b.data_ptr();
    float *out_ptr = out.data_ptr();
    for (int64_t i = 0; i < n; ++i) {
      out_ptr[i] = op(a_ptr[i], b_ptr[i]);
    }
    return out;
  }

  const int ndim = out.ndim();
  if (ndim == 0) {
    *out.data_ptr() = op(*a.data_ptr(), *b.data_ptr());
    return out;
  }

  const std::vector<int64_t> stride_a =
      bt::detail::aligned_broadcast_strides(a.shape, a.strides, out_shape);
  const std::vector<int64_t> stride_b =
      bt::detail::aligned_broadcast_strides(b.shape, b.strides, out_shape);

  recursive_apply_binary(0, ndim, out_shape, a.data_ptr(), b.data_ptr(),
                         out.data_ptr(), stride_a, stride_b, out.strides, op);
  return out;
}

/*
 * Executes a tensor-scalar elementwise operation.
 */
template <class Op> bt::Tensor binary_ts(const bt::Tensor &a, float s, Op op) {
  bt::Tensor out(a.shape);
  const int64_t n = a.numel();
  if (n == 0)
    return out;

  if (a.is_contiguous() && out.is_contiguous()) {
    const float *a_ptr = a.data_ptr();
    float *out_ptr = out.data_ptr();
    for (int64_t i = 0; i < n; ++i) {
      out_ptr[i] = op(a_ptr[i], s);
    }
    return out;
  }

  const int ndim = a.ndim();
  if (ndim == 0) {
    *out.data_ptr() = op(*a.data_ptr(), s);
    return out;
  }

  const float scalar = s;
  const std::vector<int64_t> scalar_strides(static_cast<size_t>(ndim), 0);
  recursive_apply_binary(0, ndim, a.shape, a.data_ptr(), &scalar,
                         out.data_ptr(), a.strides, scalar_strides, out.strides,
                         op);
  return out;
}

/*
 * Executes a unary elementwise operation.
 */
template <class Op> bt::Tensor unary_t(const bt::Tensor &a, Op op) {
  bt::Tensor out(a.shape);
  const int64_t n = a.numel();
  if (n == 0)
    return out;

  if (a.is_contiguous() && out.is_contiguous()) {
    const float *a_ptr = a.data_ptr();
    float *out_ptr = out.data_ptr();
    for (int64_t i = 0; i < n; ++i) {
      out_ptr[i] = op(a_ptr[i]);
    }
    return out;
  }

  const int ndim = a.ndim();
  if (ndim == 0) {
    *out.data_ptr() = op(*a.data_ptr());
    return out;
  }

  recursive_apply_unary(0, ndim, a.shape, a.data_ptr(), out.data_ptr(),
                        a.strides, out.strides, op);
  return out;
}

[[nodiscard]] bool should_record_binary(const bt::Tensor &lhs,
                                        const bt::Tensor &rhs) {
  return bt::autograd::is_grad_enabled() &&
         (lhs.requires_grad() || rhs.requires_grad());
}

[[nodiscard]] bool should_record_unary(const bt::Tensor &input) {
  return bt::autograd::is_grad_enabled() && input.requires_grad();
}

void throw_autograd_not_implemented(const std::string_view op_name) {
  std::ostringstream oss;
  oss << "Autograd support for " << op_name
      << " is not implemented yet. Milestone 1 currently supports gradients "
         "for add, mul, and sum.";
  throw std::runtime_error(oss.str());
}

class AddTensorNode final : public bt::Node {
public:
  AddTensorNode(const bt::Tensor &lhs, const bt::Tensor &rhs)
      : bt::Node({lhs, rhs}), lhs_shape_(lhs.shape), rhs_shape_(rhs.shape) {}

  [[nodiscard]] std::vector<bt::Tensor>
  backward(const bt::Tensor &out_grad) const override {
    bt::Tensor lhs_grad = bt::autograd::reduce_sum_to_shape(out_grad, lhs_shape_);
    bt::Tensor rhs_grad = bt::autograd::reduce_sum_to_shape(out_grad, rhs_shape_);
    return {lhs_grad, rhs_grad};
  }

private:
  std::vector<int64_t> lhs_shape_;
  std::vector<int64_t> rhs_shape_;
};

class AddScalarNode final : public bt::Node {
public:
  explicit AddScalarNode(const bt::Tensor &lhs) : bt::Node({lhs}) {}

  [[nodiscard]] std::vector<bt::Tensor>
  backward(const bt::Tensor &out_grad) const override {
    return {out_grad};
  }
};

class MulTensorNode final : public bt::Node {
public:
  MulTensorNode(const bt::Tensor &lhs, const bt::Tensor &rhs)
      : bt::Node({lhs, rhs}), lhs_shape_(lhs.shape), rhs_shape_(rhs.shape) {}

  [[nodiscard]] std::vector<bt::Tensor>
  backward(const bt::Tensor &out_grad) const override {
    const std::vector<bt::Tensor> &inputs = this->inputs();
    bt::Tensor lhs_grad = out_grad * inputs[1];
    bt::Tensor rhs_grad = out_grad * inputs[0];
    lhs_grad = bt::autograd::reduce_sum_to_shape(lhs_grad, lhs_shape_);
    rhs_grad = bt::autograd::reduce_sum_to_shape(rhs_grad, rhs_shape_);
    return {lhs_grad, rhs_grad};
  }

private:
  std::vector<int64_t> lhs_shape_;
  std::vector<int64_t> rhs_shape_;
};

class MulScalarNode final : public bt::Node {
public:
  MulScalarNode(const bt::Tensor &lhs, const float scalar)
      : bt::Node({lhs}), scalar_(scalar) {}

  [[nodiscard]] std::vector<bt::Tensor>
  backward(const bt::Tensor &out_grad) const override {
    return {out_grad * scalar_};
  }

private:
  float scalar_ = 1.0f;
};

} // namespace

/*
 * Namespace: bt
 * Purpose: Public BareTensor C++ API surface.
 */
namespace bt {

/*
 * Elementwise tensor-tensor addition.
 */
Tensor Tensor::operator+(const Tensor &rhs) const {
  Tensor out = binary_tt(*this, rhs, ops::Add{});
  if (should_record_binary(*this, rhs)) {
    out.set_grad_fn(std::make_shared<AddTensorNode>(*this, rhs));
  }
  return out;
}

/*
 * Elementwise tensor-scalar addition.
 */
Tensor Tensor::operator+(float rhs) const {
  Tensor out = binary_ts(*this, rhs, ops::Add{});
  if (should_record_unary(*this)) {
    out.set_grad_fn(std::make_shared<AddScalarNode>(*this));
  }
  return out;
}

/*
 * Elementwise tensor-tensor subtraction.
 */
Tensor Tensor::operator-(const Tensor &rhs) const {
  if (should_record_binary(*this, rhs)) {
    throw_autograd_not_implemented("tensor-tensor subtraction");
  }
  return binary_tt(*this, rhs, ops::Sub{});
}

/*
 * Elementwise tensor-scalar subtraction.
 */
Tensor Tensor::operator-(float rhs) const {
  if (should_record_unary(*this)) {
    throw_autograd_not_implemented("tensor-scalar subtraction");
  }
  return binary_ts(*this, rhs, ops::Sub{});
}

/*
 * Elementwise tensor-tensor multiplication.
 */
Tensor Tensor::operator*(const Tensor &rhs) const {
  Tensor out = binary_tt(*this, rhs, ops::Mul{});
  if (should_record_binary(*this, rhs)) {
    out.set_grad_fn(std::make_shared<MulTensorNode>(*this, rhs));
  }
  return out;
}

/*
 * Elementwise tensor-scalar multiplication.
 */
Tensor Tensor::operator*(float rhs) const {
  Tensor out = binary_ts(*this, rhs, ops::Mul{});
  if (should_record_unary(*this)) {
    out.set_grad_fn(std::make_shared<MulScalarNode>(*this, rhs));
  }
  return out;
}

/*
 * Elementwise tensor-tensor division.
 */
Tensor Tensor::operator/(const Tensor &rhs) const {
  if (should_record_binary(*this, rhs)) {
    throw_autograd_not_implemented("tensor-tensor division");
  }
  return binary_tt(*this, rhs, ops::Div{});
}

/*
 * Elementwise tensor-scalar division.
 */
Tensor Tensor::operator/(float rhs) const {
  if (should_record_unary(*this)) {
    throw_autograd_not_implemented("tensor-scalar division");
  }
  return binary_ts(*this, rhs, ops::Div{});
}

/*
 * Elementwise exponential.
 */
Tensor Tensor::exp() const {
  if (should_record_unary(*this)) {
    throw_autograd_not_implemented("exp");
  }
  return unary_t(*this, ops::Exp{});
}

/*
 * Elementwise natural logarithm.
 */
Tensor Tensor::log() const {
  if (should_record_unary(*this)) {
    throw_autograd_not_implemented("log");
  }
  return unary_t(*this, ops::Log{});
}

} /* namespace bt */
