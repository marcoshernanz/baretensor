/*
 * File: native/src/ops.cpp
 * Purpose: Implements elementwise tensor operations, comparisons, and where().
 */

#include "bt/ops.h"

#include <cstdint>
#include <memory>
#include <sstream>
#include <string_view>
#include <vector>

#include "bt/detail/autograd_record.h"
#include "bt/detail/broadcast.h"
#include "bt/detail/tensor_cast.h"
#include "bt/detail/tensor_copy.h"
#include "bt/detail/tensor_validation.h"
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
template <typename Lhs, typename Rhs, typename Out, class Op>
void recursive_apply_binary(const int dim, const int ndim, const std::vector<int64_t> &shape,
                            const Lhs *lhs, const Rhs *rhs, Out *out,
                            const std::vector<int64_t> &lhs_strides,
                            const std::vector<int64_t> &rhs_strides,
                            const std::vector<int64_t> &out_strides, const Op &op) {
  if (shape[static_cast<size_t>(dim)] == 0) {
    return;
  }

  if (dim == ndim - 1) {
    for (int64_t i = 0; i < shape[static_cast<size_t>(dim)]; ++i) {
      *out = static_cast<Out>(op(*lhs, *rhs));
      lhs += lhs_strides[static_cast<size_t>(dim)];
      rhs += rhs_strides[static_cast<size_t>(dim)];
      out += out_strides[static_cast<size_t>(dim)];
    }
    return;
  }

  for (int64_t i = 0; i < shape[static_cast<size_t>(dim)]; ++i) {
    recursive_apply_binary(dim + 1, ndim, shape, lhs, rhs, out, lhs_strides, rhs_strides,
                           out_strides, op);
    lhs += lhs_strides[static_cast<size_t>(dim)];
    rhs += rhs_strides[static_cast<size_t>(dim)];
    out += out_strides[static_cast<size_t>(dim)];
  }
}

/*
 * Applies a ternary select operation over strided inputs by recursively
 * traversing N-D shape space.
 */
template <typename Cond, typename Scalar>
void recursive_apply_where(const int dim, const int ndim, const std::vector<int64_t> &shape,
                           const Cond *condition, const Scalar *input, const Scalar *other,
                           Scalar *out, const std::vector<int64_t> &condition_strides,
                           const std::vector<int64_t> &input_strides,
                           const std::vector<int64_t> &other_strides,
                           const std::vector<int64_t> &out_strides) {
  if (shape[static_cast<size_t>(dim)] == 0) {
    return;
  }

  if (dim == ndim - 1) {
    for (int64_t i = 0; i < shape[static_cast<size_t>(dim)]; ++i) {
      *out = *condition ? *input : *other;
      condition += condition_strides[static_cast<size_t>(dim)];
      input += input_strides[static_cast<size_t>(dim)];
      other += other_strides[static_cast<size_t>(dim)];
      out += out_strides[static_cast<size_t>(dim)];
    }
    return;
  }

  for (int64_t i = 0; i < shape[static_cast<size_t>(dim)]; ++i) {
    recursive_apply_where(dim + 1, ndim, shape, condition, input, other, out, condition_strides,
                          input_strides, other_strides, out_strides);
    condition += condition_strides[static_cast<size_t>(dim)];
    input += input_strides[static_cast<size_t>(dim)];
    other += other_strides[static_cast<size_t>(dim)];
    out += out_strides[static_cast<size_t>(dim)];
  }
}

/*
 * Applies a unary operation over one strided input by recursively traversing
 * N-D shape space.
 */
template <class Op>
void recursive_apply_unary(const int dim, const int ndim, const std::vector<int64_t> &shape,
                           const float *input, float *out,
                           const std::vector<int64_t> &input_strides,
                           const std::vector<int64_t> &out_strides, const Op &op) {
  if (shape[static_cast<size_t>(dim)] == 0) {
    return;
  }

  if (dim == ndim - 1) {
    for (int64_t i = 0; i < shape[static_cast<size_t>(dim)]; ++i) {
      *out = op(*input);
      input += input_strides[static_cast<size_t>(dim)];
      out += out_strides[static_cast<size_t>(dim)];
    }
    return;
  }

  for (int64_t i = 0; i < shape[static_cast<size_t>(dim)]; ++i) {
    recursive_apply_unary(dim + 1, ndim, shape, input, out, input_strides, out_strides, op);
    input += input_strides[static_cast<size_t>(dim)];
    out += out_strides[static_cast<size_t>(dim)];
  }
}

/*
 * Infers the common broadcasted output shape for three operands.
 */
[[nodiscard]] std::vector<int64_t>
infer_ternary_broadcast_shape(const std::vector<int64_t> &a_shape,
                              const std::vector<int64_t> &b_shape,
                              const std::vector<int64_t> &c_shape) {
  const std::vector<int64_t> ab_shape = bt::detail::infer_broadcast_shape(a_shape, b_shape);
  return bt::detail::infer_broadcast_shape(ab_shape, c_shape);
}

/*
 * Executes a typed tensor-tensor elementwise operation with broadcasting
 * support.
 */
template <typename Lhs, typename Rhs, typename Out, class Op>
bt::Tensor binary_tt_typed(const bt::Tensor &lhs, const bt::Tensor &rhs,
                           const bt::ScalarType out_dtype, const Op &op) {
  const std::vector<int64_t> out_shape = bt::detail::infer_broadcast_shape(lhs.shape, rhs.shape);
  bt::Tensor out(out_shape, out_dtype);

  const int64_t n = out.numel();
  if (n == 0) {
    return out;
  }

  const bool no_broadcast = (lhs.shape == out_shape) && (rhs.shape == out_shape);
  if (no_broadcast && lhs.is_contiguous() && rhs.is_contiguous() && out.is_contiguous()) {
    const Lhs *lhs_ptr = lhs.data_ptr<Lhs>();
    const Rhs *rhs_ptr = rhs.data_ptr<Rhs>();
    Out *out_ptr = out.data_ptr<Out>();
    for (int64_t i = 0; i < n; ++i) {
      out_ptr[i] = static_cast<Out>(op(lhs_ptr[i], rhs_ptr[i]));
    }
    return out;
  }

  const int ndim = out.ndim();
  if (ndim == 0) {
    *out.data_ptr<Out>() = static_cast<Out>(op(*lhs.data_ptr<Lhs>(), *rhs.data_ptr<Rhs>()));
    return out;
  }

  const std::vector<int64_t> lhs_broadcast_strides =
      bt::detail::aligned_broadcast_strides(lhs.shape, lhs.strides, out_shape);
  const std::vector<int64_t> rhs_broadcast_strides =
      bt::detail::aligned_broadcast_strides(rhs.shape, rhs.strides, out_shape);

  recursive_apply_binary(0, ndim, out_shape, lhs.data_ptr<Lhs>(), rhs.data_ptr<Rhs>(),
                         out.data_ptr<Out>(), lhs_broadcast_strides, rhs_broadcast_strides,
                         out.strides, op);
  return out;
}

/*
 * Executes a typed tensor-scalar elementwise operation.
 */
template <typename Scalar, typename Out, class Op>
bt::Tensor binary_ts_typed(const bt::Tensor &input, const Scalar scalar,
                           const bt::ScalarType out_dtype, const Op &op) {
  bt::Tensor out(input.shape, out_dtype);
  const int64_t n = input.numel();
  if (n == 0) {
    return out;
  }

  if (input.is_contiguous() && out.is_contiguous()) {
    const Scalar *input_ptr = input.data_ptr<Scalar>();
    Out *out_ptr = out.data_ptr<Out>();
    for (int64_t i = 0; i < n; ++i) {
      out_ptr[i] = static_cast<Out>(op(input_ptr[i], scalar));
    }
    return out;
  }

  const int ndim = input.ndim();
  if (ndim == 0) {
    *out.data_ptr<Out>() = static_cast<Out>(op(*input.data_ptr<Scalar>(), scalar));
    return out;
  }

  const std::vector<int64_t> scalar_strides(static_cast<size_t>(ndim), 0);
  recursive_apply_binary(0, ndim, input.shape, input.data_ptr<Scalar>(), &scalar,
                         out.data_ptr<Out>(), input.strides, scalar_strides, out.strides, op);
  return out;
}

/*
 * Executes a float32 tensor-tensor arithmetic operation.
 */
template <class Op>
bt::Tensor binary_tt_float(const bt::Tensor &lhs, const bt::Tensor &rhs, const Op &op) {
  bt::detail::ensure_same_dtype(lhs, rhs, "elementwise operation");
  bt::detail::ensure_float32(lhs, "elementwise operation", "lhs");
  return binary_tt_typed<float, float, float>(lhs, rhs, bt::ScalarType::kFloat32, op);
}

/*
 * Executes a float32 tensor-scalar arithmetic operation.
 */
template <class Op>
bt::Tensor binary_ts_float(const bt::Tensor &input, const float scalar, const Op &op) {
  bt::detail::ensure_float32(input, "elementwise scalar operation");
  return binary_ts_typed<float, float>(input, scalar, bt::ScalarType::kFloat32, op);
}

/*
 * Executes a tensor-tensor comparison, returning a bool tensor.
 */
template <class Op>
bt::Tensor compare_tt(const bt::Tensor &lhs, const bt::Tensor &rhs, const Op &op) {
  bt::detail::ensure_same_dtype(lhs, rhs, "comparison");
  return bt::visit_dtype(lhs.dtype(), [&]<typename Scalar>() {
    return binary_tt_typed<Scalar, Scalar, bool>(lhs, rhs, bt::ScalarType::kBool, op);
  });
}

/*
 * Executes a tensor-scalar comparison, returning a bool tensor.
 */
template <class Op>
bt::Tensor compare_ts(const bt::Tensor &input, const double scalar, const Op &op) {
  return bt::visit_dtype(input.dtype(), [&]<typename Scalar>() {
    const Scalar typed_scalar =
        bt::detail::cast_tensor_scalar<double, Scalar>(scalar, "comparison");
    return binary_ts_typed<Scalar, bool>(input, typed_scalar, bt::ScalarType::kBool, op);
  });
}

/*
 * Executes a typed where() kernel with three-way broadcasting.
 */
template <typename Scalar>
bt::Tensor where_typed(const bt::Tensor &condition, const bt::Tensor &input,
                       const bt::Tensor &other) {
  const std::vector<int64_t> out_shape =
      infer_ternary_broadcast_shape(condition.shape, input.shape, other.shape);
  bt::Tensor out(out_shape, input.dtype());

  const int64_t n = out.numel();
  if (n == 0) {
    return out;
  }

  const bool no_broadcast =
      (condition.shape == out_shape) && (input.shape == out_shape) && (other.shape == out_shape);
  if (no_broadcast && condition.is_contiguous() && input.is_contiguous() && other.is_contiguous() &&
      out.is_contiguous()) {
    const bool *condition_ptr = condition.data_ptr<bool>();
    const Scalar *input_ptr = input.data_ptr<Scalar>();
    const Scalar *other_ptr = other.data_ptr<Scalar>();
    Scalar *out_ptr = out.data_ptr<Scalar>();
    for (int64_t i = 0; i < n; ++i) {
      out_ptr[i] = condition_ptr[i] ? input_ptr[i] : other_ptr[i];
    }
    return out;
  }

  const int ndim = out.ndim();
  if (ndim == 0) {
    *out.data_ptr<Scalar>() =
        *condition.data_ptr<bool>() ? *input.data_ptr<Scalar>() : *other.data_ptr<Scalar>();
    return out;
  }

  const std::vector<int64_t> condition_broadcast_strides =
      bt::detail::aligned_broadcast_strides(condition.shape, condition.strides, out_shape);
  const std::vector<int64_t> input_broadcast_strides =
      bt::detail::aligned_broadcast_strides(input.shape, input.strides, out_shape);
  const std::vector<int64_t> other_broadcast_strides =
      bt::detail::aligned_broadcast_strides(other.shape, other.strides, out_shape);

  recursive_apply_where(0, ndim, out_shape, condition.data_ptr<bool>(), input.data_ptr<Scalar>(),
                        other.data_ptr<Scalar>(), out.data_ptr<Scalar>(),
                        condition_broadcast_strides, input_broadcast_strides,
                        other_broadcast_strides, out.strides);
  return out;
}

/*
 * Executes a unary float32 elementwise operation.
 */
template <class Op> bt::Tensor unary_t(const bt::Tensor &a, const Op &op) {
  bt::detail::ensure_float32(a, "unary operation");
  bt::Tensor out(a.shape);
  const int64_t n = a.numel();
  if (n == 0) {
    return out;
  }

  if (a.is_contiguous() && out.is_contiguous()) {
    const float *a_ptr = a.data_ptr<float>();
    float *out_ptr = out.data_ptr<float>();
    for (int64_t i = 0; i < n; ++i) {
      out_ptr[i] = op(a_ptr[i]);
    }
    return out;
  }

  const int ndim = a.ndim();
  if (ndim == 0) {
    *out.data_ptr<float>() = op(*a.data_ptr<float>());
    return out;
  }

  recursive_apply_unary(0, ndim, a.shape, a.data_ptr<float>(), out.data_ptr<float>(), a.strides,
                        out.strides, op);
  return out;
}

/*
 * Executes an out-of-place elementwise computation and copies the result back
 * into lhs storage. This operation is only valid when gradient recording is
 * disabled.
 */
template <class Compute>
bt::Tensor &apply_inplace(bt::Tensor &lhs, const char *operation_name, const Compute &compute) {
  if (bt::autograd::is_grad_enabled()) {
    throw std::runtime_error(std::string(operation_name) +
                             " is only supported inside bt.no_grad().");
  }

  bt::detail::validate_copy_metadata(lhs, operation_name);
  const bt::Tensor out = compute();
  bt::detail::validate_copy_metadata(out, operation_name);

  if (out.shape != lhs.shape) {
    std::ostringstream oss;
    oss << operation_name << " cannot change tensor shape from "
        << bt::detail::shape_to_string(lhs.shape) << " to "
        << bt::detail::shape_to_string(out.shape) << ".";
    throw std::invalid_argument(oss.str());
  }

  bt::detail::copy_tensor_values(out, lhs);
  return lhs;
}

class AddTensorNode final : public bt::Node {
public:
  AddTensorNode(const bt::Tensor &lhs, const bt::Tensor &rhs)
      : bt::Node({lhs, rhs}), lhs_shape_(lhs.shape), rhs_shape_(rhs.shape) {}

  [[nodiscard]] std::vector<bt::Tensor> backward(const bt::Tensor &out_grad) const override {
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

  [[nodiscard]] std::vector<bt::Tensor> backward(const bt::Tensor &out_grad) const override {
    return {out_grad};
  }
};

class MulTensorNode final : public bt::Node {
public:
  MulTensorNode(const bt::Tensor &lhs, const bt::Tensor &rhs)
      : bt::Node({lhs, rhs}), lhs_shape_(lhs.shape), rhs_shape_(rhs.shape) {}

  [[nodiscard]] std::vector<bt::Tensor> backward(const bt::Tensor &out_grad) const override {
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
  MulScalarNode(const bt::Tensor &lhs, const float scalar) : bt::Node({lhs}), scalar_(scalar) {}

  [[nodiscard]] std::vector<bt::Tensor> backward(const bt::Tensor &out_grad) const override {
    return {out_grad * scalar_};
  }

private:
  float scalar_ = 1.0f;
};

class SubTensorNode final : public bt::Node {
public:
  SubTensorNode(const bt::Tensor &lhs, const bt::Tensor &rhs)
      : bt::Node({lhs, rhs}), lhs_shape_(lhs.shape), rhs_shape_(rhs.shape) {}

  [[nodiscard]] std::vector<bt::Tensor> backward(const bt::Tensor &out_grad) const override {
    bt::Tensor lhs_grad = bt::autograd::reduce_sum_to_shape(out_grad, lhs_shape_);
    bt::Tensor rhs_grad = bt::autograd::reduce_sum_to_shape(out_grad * -1.0f, rhs_shape_);
    return {lhs_grad, rhs_grad};
  }

private:
  std::vector<int64_t> lhs_shape_;
  std::vector<int64_t> rhs_shape_;
};

class SubScalarNode final : public bt::Node {
public:
  explicit SubScalarNode(const bt::Tensor &lhs) : bt::Node({lhs}) {}

  [[nodiscard]] std::vector<bt::Tensor> backward(const bt::Tensor &out_grad) const override {
    return {out_grad};
  }
};

class DivTensorNode final : public bt::Node {
public:
  DivTensorNode(const bt::Tensor &lhs, const bt::Tensor &rhs)
      : bt::Node({lhs, rhs}), lhs_shape_(lhs.shape), rhs_shape_(rhs.shape) {}

  [[nodiscard]] std::vector<bt::Tensor> backward(const bt::Tensor &out_grad) const override {
    const std::vector<bt::Tensor> &inputs = this->inputs();
    const bt::Tensor rhs_sq = inputs[1] * inputs[1];

    bt::Tensor lhs_grad = out_grad / inputs[1];
    bt::Tensor rhs_grad = (out_grad * inputs[0]) / rhs_sq;
    rhs_grad = rhs_grad * -1.0f;

    lhs_grad = bt::autograd::reduce_sum_to_shape(lhs_grad, lhs_shape_);
    rhs_grad = bt::autograd::reduce_sum_to_shape(rhs_grad, rhs_shape_);
    return {lhs_grad, rhs_grad};
  }

private:
  std::vector<int64_t> lhs_shape_;
  std::vector<int64_t> rhs_shape_;
};

class DivScalarNode final : public bt::Node {
public:
  DivScalarNode(const bt::Tensor &lhs, const float scalar) : bt::Node({lhs}), scalar_(scalar) {}

  [[nodiscard]] std::vector<bt::Tensor> backward(const bt::Tensor &out_grad) const override {
    return {out_grad / scalar_};
  }

private:
  float scalar_ = 1.0f;
};

class WhereNode final : public bt::Node {
public:
  WhereNode(const bt::Tensor &condition, const bt::Tensor &input, const bt::Tensor &other)
      : bt::Node({input, other}), condition_(condition), input_shape_(input.shape),
        other_shape_(other.shape) {}

  [[nodiscard]] std::vector<bt::Tensor> backward(const bt::Tensor &out_grad) const override {
    const bt::Tensor zero = bt::zeros(out_grad.shape, out_grad.dtype());
    bt::Tensor input_grad = bt::where(condition_, out_grad, zero);
    bt::Tensor other_grad = bt::where(condition_, zero, out_grad);
    input_grad = bt::autograd::reduce_sum_to_shape(input_grad, input_shape_);
    other_grad = bt::autograd::reduce_sum_to_shape(other_grad, other_shape_);
    return {input_grad, other_grad};
  }

private:
  bt::Tensor condition_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> other_shape_;
};

class ExpNode final : public bt::Node {
public:
  explicit ExpNode(const bt::Tensor &input) : bt::Node({input}) {}

  [[nodiscard]] std::vector<bt::Tensor> backward(const bt::Tensor &out_grad) const override {
    const bt::Tensor &input = this->inputs()[0];
    return {out_grad * input.exp()};
  }
};

class LogNode final : public bt::Node {
public:
  explicit LogNode(const bt::Tensor &input) : bt::Node({input}) {}

  [[nodiscard]] std::vector<bt::Tensor> backward(const bt::Tensor &out_grad) const override {
    const bt::Tensor &input = this->inputs()[0];
    return {out_grad / input};
  }
};

class TanhNode final : public bt::Node {
public:
  explicit TanhNode(const bt::Tensor &input) : bt::Node({input}) {}

  [[nodiscard]] std::vector<bt::Tensor> backward(const bt::Tensor &out_grad) const override {
    const bt::Tensor tanh_input = this->inputs()[0].tanh();
    const bt::Tensor tanh_sq = tanh_input * tanh_input;
    return {out_grad * ((tanh_sq * -1.0f) + 1.0f)};
  }
};

class SigmoidNode final : public bt::Node {
public:
  explicit SigmoidNode(const bt::Tensor &input) : bt::Node({input}) {}

  [[nodiscard]] std::vector<bt::Tensor> backward(const bt::Tensor &out_grad) const override {
    const bt::Tensor sigmoid_input = this->inputs()[0].sigmoid();
    return {out_grad * sigmoid_input * (1.0f - sigmoid_input)};
  }
};

} // namespace

/*
 * Namespace: bt
 * Purpose: Public BareTensor C++ API surface.
 */
namespace bt {

/*
 * Elementwise unary negation.
 */
Tensor Tensor::operator-() const { return (*this) * -1.0f; }

/*
 * Elementwise scalar-tensor addition.
 */
Tensor operator+(const float lhs, const Tensor &rhs) { return rhs + lhs; }

/*
 * Elementwise scalar-tensor subtraction.
 */
Tensor operator-(const float lhs, const Tensor &rhs) { return (-rhs) + lhs; }

/*
 * Elementwise scalar-tensor multiplication.
 */
Tensor operator*(const float lhs, const Tensor &rhs) { return rhs * lhs; }

/*
 * Elementwise scalar-tensor division.
 */
Tensor operator/(const float lhs, const Tensor &rhs) { return bt::full(rhs.shape, lhs) / rhs; }

/*
 * Selects values from input and other based on a boolean condition.
 */
Tensor where(const Tensor &condition, const Tensor &input, const Tensor &other) {
  bt::detail::ensure_bool(condition, "where()", "condition");
  bt::detail::ensure_same_dtype(input, other, "where()");

  Tensor out = bt::visit_dtype(input.dtype(), [&]<typename Scalar>() {
    return where_typed<Scalar>(condition, input, other);
  });
  if (bt::detail::should_record_binary(input, other)) {
    out.set_grad_fn(std::make_shared<WhereNode>(condition, input, other));
  }
  return out;
}

/*
 * Elementwise tensor-tensor addition.
 */
Tensor Tensor::operator+(const Tensor &rhs) const {
  Tensor out = binary_tt_float(*this, rhs, ops::Add{});
  if (bt::detail::should_record_binary(*this, rhs)) {
    out.set_grad_fn(std::make_shared<AddTensorNode>(*this, rhs));
  }
  return out;
}

/*
 * Elementwise tensor-scalar addition.
 */
Tensor Tensor::operator+(const float rhs) const {
  Tensor out = binary_ts_float(*this, rhs, ops::Add{});
  if (bt::detail::should_record_unary(*this)) {
    out.set_grad_fn(std::make_shared<AddScalarNode>(*this));
  }
  return out;
}

/*
 * In-place tensor-tensor addition.
 */
Tensor &Tensor::operator+=(const Tensor &rhs) {
  return apply_inplace(*this, "__iadd__()", [this, &rhs]() { return *this + rhs; });
}

/*
 * In-place tensor-scalar addition.
 */
Tensor &Tensor::operator+=(const float rhs) {
  return apply_inplace(*this, "__iadd__()", [this, rhs]() { return *this + rhs; });
}

/*
 * Elementwise tensor-tensor subtraction.
 */
Tensor Tensor::operator-(const Tensor &rhs) const {
  Tensor out = binary_tt_float(*this, rhs, ops::Sub{});
  if (bt::detail::should_record_binary(*this, rhs)) {
    out.set_grad_fn(std::make_shared<SubTensorNode>(*this, rhs));
  }
  return out;
}

/*
 * Elementwise tensor-scalar subtraction.
 */
Tensor Tensor::operator-(const float rhs) const {
  Tensor out = binary_ts_float(*this, rhs, ops::Sub{});
  if (bt::detail::should_record_unary(*this)) {
    out.set_grad_fn(std::make_shared<SubScalarNode>(*this));
  }
  return out;
}

/*
 * In-place tensor-tensor subtraction.
 */
Tensor &Tensor::operator-=(const Tensor &rhs) {
  return apply_inplace(*this, "__isub__()", [this, &rhs]() { return *this - rhs; });
}

/*
 * In-place tensor-scalar subtraction.
 */
Tensor &Tensor::operator-=(const float rhs) {
  return apply_inplace(*this, "__isub__()", [this, rhs]() { return *this - rhs; });
}

/*
 * Elementwise tensor-tensor multiplication.
 */
Tensor Tensor::operator*(const Tensor &rhs) const {
  Tensor out = binary_tt_float(*this, rhs, ops::Mul{});
  if (bt::detail::should_record_binary(*this, rhs)) {
    out.set_grad_fn(std::make_shared<MulTensorNode>(*this, rhs));
  }
  return out;
}

/*
 * Elementwise tensor-scalar multiplication.
 */
Tensor Tensor::operator*(const float rhs) const {
  Tensor out = binary_ts_float(*this, rhs, ops::Mul{});
  if (bt::detail::should_record_unary(*this)) {
    out.set_grad_fn(std::make_shared<MulScalarNode>(*this, rhs));
  }
  return out;
}

/*
 * In-place tensor-tensor multiplication.
 */
Tensor &Tensor::operator*=(const Tensor &rhs) {
  return apply_inplace(*this, "__imul__()", [this, &rhs]() { return *this * rhs; });
}

/*
 * In-place tensor-scalar multiplication.
 */
Tensor &Tensor::operator*=(const float rhs) {
  return apply_inplace(*this, "__imul__()", [this, rhs]() { return *this * rhs; });
}

/*
 * Elementwise tensor-tensor division.
 */
Tensor Tensor::operator/(const Tensor &rhs) const {
  Tensor out = binary_tt_float(*this, rhs, ops::Div{});
  if (bt::detail::should_record_binary(*this, rhs)) {
    out.set_grad_fn(std::make_shared<DivTensorNode>(*this, rhs));
  }
  return out;
}

/*
 * Elementwise tensor-scalar division.
 */
Tensor Tensor::operator/(const float rhs) const {
  Tensor out = binary_ts_float(*this, rhs, ops::Div{});
  if (bt::detail::should_record_unary(*this)) {
    out.set_grad_fn(std::make_shared<DivScalarNode>(*this, rhs));
  }
  return out;
}

/*
 * In-place tensor-tensor division.
 */
Tensor &Tensor::operator/=(const Tensor &rhs) {
  return apply_inplace(*this, "__itruediv__()", [this, &rhs]() { return *this / rhs; });
}

/*
 * In-place tensor-scalar division.
 */
Tensor &Tensor::operator/=(const float rhs) {
  return apply_inplace(*this, "__itruediv__()", [this, rhs]() { return *this / rhs; });
}

/*
 * Elementwise tensor-tensor equality comparison.
 */
Tensor Tensor::operator==(const Tensor &rhs) const { return compare_tt(*this, rhs, ops::Eq{}); }

/*
 * Elementwise tensor-scalar equality comparison.
 */
Tensor Tensor::operator==(const float rhs) const { return compare_ts(*this, rhs, ops::Eq{}); }

/*
 * Elementwise tensor-tensor inequality comparison.
 */
Tensor Tensor::operator!=(const Tensor &rhs) const { return compare_tt(*this, rhs, ops::Ne{}); }

/*
 * Elementwise tensor-scalar inequality comparison.
 */
Tensor Tensor::operator!=(const float rhs) const { return compare_ts(*this, rhs, ops::Ne{}); }

/*
 * Elementwise tensor-tensor less-than comparison.
 */
Tensor Tensor::operator<(const Tensor &rhs) const { return compare_tt(*this, rhs, ops::Lt{}); }

/*
 * Elementwise tensor-scalar less-than comparison.
 */
Tensor Tensor::operator<(const float rhs) const { return compare_ts(*this, rhs, ops::Lt{}); }

/*
 * Elementwise tensor-tensor less-than-or-equal comparison.
 */
Tensor Tensor::operator<=(const Tensor &rhs) const { return compare_tt(*this, rhs, ops::Le{}); }

/*
 * Elementwise tensor-scalar less-than-or-equal comparison.
 */
Tensor Tensor::operator<=(const float rhs) const { return compare_ts(*this, rhs, ops::Le{}); }

/*
 * Elementwise tensor-tensor greater-than comparison.
 */
Tensor Tensor::operator>(const Tensor &rhs) const { return compare_tt(*this, rhs, ops::Gt{}); }

/*
 * Elementwise tensor-scalar greater-than comparison.
 */
Tensor Tensor::operator>(const float rhs) const { return compare_ts(*this, rhs, ops::Gt{}); }

/*
 * Elementwise tensor-tensor greater-than-or-equal comparison.
 */
Tensor Tensor::operator>=(const Tensor &rhs) const { return compare_tt(*this, rhs, ops::Ge{}); }

/*
 * Elementwise tensor-scalar greater-than-or-equal comparison.
 */
Tensor Tensor::operator>=(const float rhs) const { return compare_ts(*this, rhs, ops::Ge{}); }

/*
 * Elementwise exponential.
 */
Tensor Tensor::exp() const {
  Tensor out = unary_t(*this, ops::Exp{});
  if (bt::detail::should_record_unary(*this)) {
    out.set_grad_fn(std::make_shared<ExpNode>(*this));
  }
  return out;
}

/*
 * Elementwise natural logarithm.
 */
Tensor Tensor::log() const {
  Tensor out = unary_t(*this, ops::Log{});
  if (bt::detail::should_record_unary(*this)) {
    out.set_grad_fn(std::make_shared<LogNode>(*this));
  }
  return out;
}

/*
 * Elementwise hyperbolic tangent.
 */
Tensor Tensor::tanh() const {
  Tensor out = unary_t(*this, ops::Tanh{});
  if (bt::detail::should_record_unary(*this)) {
    out.set_grad_fn(std::make_shared<TanhNode>(*this));
  }
  return out;
}

/*
 * Elementwise logistic sigmoid.
 */
Tensor Tensor::sigmoid() const {
  Tensor out = unary_t(*this, ops::Sigmoid{});
  if (bt::detail::should_record_unary(*this)) {
    out.set_grad_fn(std::make_shared<SigmoidNode>(*this));
  }
  return out;
}

} /* namespace bt */
