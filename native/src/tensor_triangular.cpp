/*
 * File: native/src/tensor_triangular.cpp
 * Purpose: Implements triangular tensor ops and their autograd node.
 */

#include "bt/tensor.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "bt/detail/autograd_record.h"
#include "bt/detail/format.h"
#include "bt/detail/tensor_validation.h"

/*
 * Namespace: (anonymous)
 * Purpose: Private implementation details local to this translation unit.
 */
namespace {

enum class TriangularMode { kUpper, kLower };

[[nodiscard]] int64_t saturating_add_int64(const int64_t lhs, const int64_t rhs) {
  if (rhs > 0 && lhs > (std::numeric_limits<int64_t>::max() - rhs)) {
    return std::numeric_limits<int64_t>::max();
  }
  if (rhs < 0 && lhs < (std::numeric_limits<int64_t>::lowest() - rhs)) {
    return std::numeric_limits<int64_t>::lowest();
  }
  return lhs + rhs;
}

void validate_triangular_input(const bt::Tensor &input, const std::string_view operation_name) {
  const std::string op_name(operation_name);
  bt::detail::validate_copy_metadata(input, op_name);

  if (input.ndim() >= 2) {
    return;
  }

  std::ostringstream oss;
  oss << operation_name << " failed for tensor with shape "
      << bt::detail::shape_to_string(input.shape) << ": expected ndim() >= 2, but got "
      << input.ndim() << ".";
  throw std::invalid_argument(oss.str());
}

template <typename T>
void copy_triangular_matrix(const T *input_matrix, T *output_matrix, const int64_t rows,
                            const int64_t cols, const int64_t input_row_stride,
                            const int64_t input_col_stride, const int64_t output_row_stride,
                            const int64_t output_col_stride, const int64_t diagonal,
                            const TriangularMode mode) {
  for (int64_t row = 0; row < rows; ++row) {
    const int64_t boundary = saturating_add_int64(row, diagonal);
    int64_t col_start = 0;
    int64_t col_end = 0;

    if (mode == TriangularMode::kUpper) {
      col_start = std::max<int64_t>(0, boundary);
      col_end = cols;
    } else {
      col_start = 0;
      col_end = std::min<int64_t>(cols, saturating_add_int64(boundary, 1));
    }

    if (col_start >= col_end) {
      continue;
    }

    const T *input_ptr = input_matrix + (row * input_row_stride) + (col_start * input_col_stride);
    T *output_ptr = output_matrix + (row * output_row_stride) + (col_start * output_col_stride);
    for (int64_t col = col_start; col < col_end; ++col) {
      *output_ptr = *input_ptr;
      input_ptr += input_col_stride;
      output_ptr += output_col_stride;
    }
  }
}

template <typename T>
void copy_triangular_batch(const size_t batch_dim, const size_t batch_ndim,
                           const std::vector<int64_t> &shape, const T *input_ptr, T *output_ptr,
                           const std::vector<int64_t> &input_strides,
                           const std::vector<int64_t> &output_strides,
                           const int64_t input_row_stride, const int64_t input_col_stride,
                           const int64_t output_row_stride, const int64_t output_col_stride,
                           const int64_t rows, const int64_t cols, const int64_t diagonal,
                           const TriangularMode mode) {
  if (batch_dim == batch_ndim) {
    copy_triangular_matrix(input_ptr, output_ptr, rows, cols, input_row_stride, input_col_stride,
                           output_row_stride, output_col_stride, diagonal, mode);
    return;
  }

  const int64_t dim_size = shape[batch_dim];
  for (int64_t idx = 0; idx < dim_size; ++idx) {
    copy_triangular_batch(batch_dim + 1, batch_ndim, shape,
                          input_ptr + (idx * input_strides[batch_dim]),
                          output_ptr + (idx * output_strides[batch_dim]), input_strides,
                          output_strides, input_row_stride, input_col_stride, output_row_stride,
                          output_col_stride, rows, cols, diagonal, mode);
  }
}

class TriangularNode final : public bt::Node {
public:
  TriangularNode(const bt::Tensor &input, const int64_t diagonal, const TriangularMode mode)
      : bt::Node({input}), diagonal_(diagonal), mode_(mode) {}

  [[nodiscard]] std::vector<bt::Tensor> backward(const bt::Tensor &out_grad) const override {
    if (mode_ == TriangularMode::kUpper) {
      return {out_grad.triu(diagonal_)};
    }
    return {out_grad.tril(diagonal_)};
  }

private:
  int64_t diagonal_ = 0;
  TriangularMode mode_ = TriangularMode::kUpper;
};

[[nodiscard]] bt::Tensor triangular_impl(const bt::Tensor &input, const int64_t diagonal,
                                         const TriangularMode mode,
                                         const std::string_view operation_name) {
  validate_triangular_input(input, operation_name);

  bt::Tensor output(input.shape, input.dtype());
  const size_t rank = input.shape.size();
  const size_t batch_ndim = rank - 2;
  const int64_t rows = input.shape[rank - 2];
  const int64_t cols = input.shape[rank - 1];
  const int64_t input_row_stride = input.strides[rank - 2];
  const int64_t input_col_stride = input.strides[rank - 1];
  const int64_t output_row_stride = output.strides[rank - 2];
  const int64_t output_col_stride = output.strides[rank - 1];

  bt::visit_dtype(input.dtype(), [&]<typename T>() {
    const T *input_ptr = input.data_ptr<T>();
    T *output_ptr = output.data_ptr<T>();

    if (batch_ndim == 0) {
      copy_triangular_matrix(input_ptr, output_ptr, rows, cols, input_row_stride, input_col_stride,
                             output_row_stride, output_col_stride, diagonal, mode);
      return;
    }

    copy_triangular_batch(0, batch_ndim, input.shape, input_ptr, output_ptr, input.strides,
                          output.strides, input_row_stride, input_col_stride, output_row_stride,
                          output_col_stride, rows, cols, diagonal, mode);
  });

  if (bt::detail::should_record_unary(input)) {
    output.set_grad_fn(std::make_shared<TriangularNode>(input, diagonal, mode));
  }
  return output;
}

} // namespace

/*
 * Namespace: bt
 * Purpose: Public BareTensor C++ API surface.
 */
namespace bt {

Tensor Tensor::triu(const int64_t diagonal) const {
  return triangular_impl(*this, diagonal, TriangularMode::kUpper, "triu");
}

Tensor Tensor::tril(const int64_t diagonal) const {
  return triangular_impl(*this, diagonal, TriangularMode::kLower, "tril");
}

Tensor triu(const Tensor &input, const int64_t diagonal) { return input.triu(diagonal); }

Tensor tril(const Tensor &input, const int64_t diagonal) { return input.tril(diagonal); }

} // namespace bt
