/*
 * File: native/src/tensor_linalg.cpp
 * Purpose: Implements tensor linear algebra ops and their autograd nodes.
 */

#include "bt/tensor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "bt/detail/autograd_record.h"
#include "bt/detail/broadcast.h"
#include "bt/detail/format.h"
#include "bt/detail/tensor_validation.h"

/*
 * Namespace: (anonymous)
 * Purpose: Private implementation details local to this translation unit.
 */
namespace {

/*
 * Struct: MatmulCanonicalInput
 * Purpose: Stores canonical matmul metadata for one operand.
 */
struct MatmulCanonicalInput {
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  bool was_1d = false;
};

/*
 * Struct: MatmulKernelParams
 * Purpose: Bundles strides and dimensions for one matmul invocation.
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
[[nodiscard]] MatmulCanonicalInput canonicalize_matmul_input(const bt::Tensor &tensor,
                                                             const bool prepend_for_1d) {
  if (tensor.ndim() != 1) {
    return MatmulCanonicalInput{tensor.shape, tensor.strides, false};
  }

  if (prepend_for_1d) {
    return MatmulCanonicalInput{{1, tensor.shape[0]}, {0, tensor.strides[0]}, true};
  }

  return MatmulCanonicalInput{{tensor.shape[0], 1}, {tensor.strides[0], 0}, true};
}

/*
 * Removes temporary singleton dimensions introduced by 1-D promotion.
 */
[[nodiscard]] std::vector<int64_t>
matmul_result_shape(const std::vector<int64_t> &full_shape, const bool lhs_was_1d,
                    const bool rhs_was_1d) {
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
void recursive_batched_matmul(const size_t dim, const std::vector<int64_t> &batch_shape,
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
    recursive_batched_matmul(dim + 1, batch_shape, lhs, rhs, out, lhs_batch_strides,
                             rhs_batch_strides, out_batch_strides, params);
    lhs += lhs_batch_strides[dim];
    rhs += rhs_batch_strides[dim];
    out += out_batch_strides[dim];
  }
}

/*
 * Class: MatmulNode
 * Purpose: Backward pass for Tensor::matmul.
 */
class MatmulNode final : public bt::Node {
public:
  MatmulNode(const bt::Tensor &lhs, const bt::Tensor &rhs) : bt::Node({lhs, rhs}) {}

  [[nodiscard]] std::vector<bt::Tensor>
  backward(const bt::Tensor &out_grad) const override {
    const std::vector<bt::Tensor> &inputs = this->inputs();
    const bt::Tensor &lhs = inputs[0];
    const bt::Tensor &rhs = inputs[1];

    const MatmulCanonicalInput lhs_canonical_meta = canonicalize_matmul_input(lhs, true);
    const MatmulCanonicalInput rhs_canonical_meta = canonicalize_matmul_input(rhs, false);

    const bt::Tensor lhs_canonical(lhs.storage, lhs.storage_offset,
                                   lhs_canonical_meta.shape, lhs_canonical_meta.strides);
    const bt::Tensor rhs_canonical(rhs.storage, rhs.storage_offset,
                                   rhs_canonical_meta.shape, rhs_canonical_meta.strides);

    const std::vector<int64_t> lhs_batch_shape(lhs_canonical_meta.shape.begin(),
                                               lhs_canonical_meta.shape.end() - 2);
    const std::vector<int64_t> rhs_batch_shape(rhs_canonical_meta.shape.begin(),
                                               rhs_canonical_meta.shape.end() - 2);
    const std::vector<int64_t> batch_shape =
        bt::detail::infer_broadcast_shape(lhs_batch_shape, rhs_batch_shape);

    std::vector<int64_t> full_out_shape = batch_shape;
    full_out_shape.push_back(
        lhs_canonical_meta.shape[lhs_canonical_meta.shape.size() - 2]);
    full_out_shape.push_back(
        rhs_canonical_meta.shape[rhs_canonical_meta.shape.size() - 1]);

    const bt::Tensor out_grad_canonical = out_grad.reshape(full_out_shape);

    bt::Tensor lhs_grad_canonical = out_grad_canonical.matmul(rhs_canonical.mT());
    bt::Tensor rhs_grad_canonical = lhs_canonical.mT().matmul(out_grad_canonical);

    bt::Tensor lhs_grad =
        bt::autograd::reduce_sum_to_shape(lhs_grad_canonical, lhs_canonical_meta.shape);
    bt::Tensor rhs_grad =
        bt::autograd::reduce_sum_to_shape(rhs_grad_canonical, rhs_canonical_meta.shape);

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
 * Returns the matrix product of this tensor and tensor2 using matmul
 * semantics equivalent to PyTorch for dense tensors.
 */
Tensor Tensor::matmul(const Tensor &tensor2) const {
  bt::detail::validate_copy_metadata(*this, "matmul");
  bt::detail::validate_copy_metadata(tensor2, "matmul");

  if (ndim() == 0 || tensor2.ndim() == 0) {
    std::ostringstream oss;
    oss << "matmul failed for tensors with " << matmul_shapes_to_string(*this, tensor2)
        << ": both tensors must be at least 1-D, but got " << ndim() << "-D and "
        << tensor2.ndim() << "-D.";
    throw std::invalid_argument(oss.str());
  }

  const MatmulCanonicalInput lhs = canonicalize_matmul_input(*this, true);
  const MatmulCanonicalInput rhs = canonicalize_matmul_input(tensor2, false);

  const int64_t lhs_k = lhs.shape[lhs.shape.size() - 1];
  const int64_t rhs_k = rhs.shape[rhs.shape.size() - 2];
  if (lhs_k != rhs_k) {
    std::ostringstream oss;
    oss << "matmul failed for tensors with " << matmul_shapes_to_string(*this, tensor2)
        << ": inner dimensions must match (lhs.shape[-1] == rhs.shape[-2]), got " << lhs_k
        << " and " << rhs_k << ".";
    throw std::invalid_argument(oss.str());
  }

  const std::vector<int64_t> lhs_batch_shape(lhs.shape.begin(), lhs.shape.end() - 2);
  const std::vector<int64_t> rhs_batch_shape(rhs.shape.begin(), rhs.shape.end() - 2);

  std::vector<int64_t> batch_shape;
  try {
    batch_shape = detail::infer_broadcast_shape(lhs_batch_shape, rhs_batch_shape);
  } catch (const std::invalid_argument &err) {
    std::ostringstream oss;
    oss << "matmul failed for tensors with " << matmul_shapes_to_string(*this, tensor2)
        << ": batch dimensions are not broadcastable: " << err.what();
    throw std::invalid_argument(oss.str());
  }

  const std::vector<int64_t> lhs_batch_strides(lhs.strides.begin(),
                                               lhs.strides.end() - 2);
  const std::vector<int64_t> rhs_batch_strides(rhs.strides.begin(),
                                               rhs.strides.end() - 2);

  const std::vector<int64_t> lhs_batch_broadcast_strides =
      detail::aligned_broadcast_strides(lhs_batch_shape, lhs_batch_strides, batch_shape);
  const std::vector<int64_t> rhs_batch_broadcast_strides =
      detail::aligned_broadcast_strides(rhs_batch_shape, rhs_batch_strides, batch_shape);

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
                             rhs_batch_broadcast_strides, out_batch_strides, params);
  }

  const std::vector<int64_t> out_shape =
      matmul_result_shape(full_out_shape, lhs.was_1d, rhs.was_1d);
  Tensor out = out_full.reshape(out_shape);
  if (bt::detail::should_record_binary(*this, tensor2)) {
    out.set_grad_fn(std::make_shared<MatmulNode>(*this, tensor2));
  }
  return out;
}

} /* namespace bt */
