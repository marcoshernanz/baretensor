#include <vector>

template <class Op>
void recursive_apply_tt(int dim, int last_dim,
                        const std::vector<int64_t>& shape,
                        const std::vector<float>& a,
                        const std::vector<float>& b, std::vector<float>& out,
                        int64_t offset_a, int64_t offset_b, int64_t offset_out,
                        const std::vector<int64_t>& stride_a,
                        const std::vector<int64_t>& stride_b,
                        const std::vector<int64_t>& stride_out, Op op) {
  if (shape.size() == 0) {
    out[offset_out] = op(a[offset_a], b[offset_b]);
    return;
  }

  if (dim == last_dim) {
    for (int64_t i = 0; i < shape[dim]; i++) {
      out[offset_out] = op(a[offset_a], b[offset_b]);
      offset_a += stride_a[dim];
      offset_b += stride_b[dim];
      offset_out += stride_out[dim];
    }
    return;
  }

  for (int64_t i = 0; i < shape[dim]; i++) {
    recursive_apply(dim + 1, last_dim, shape, a, b, out, offset_a, offset_b,
                    offset_out, stride_a, stride_b, stride_out, op);
    offset_a += stride_a[dim];
    offset_b += stride_b[dim];
    offset_out += stride_out[dim];
  }
}