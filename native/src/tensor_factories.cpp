/*
 * File: native/src/tensor_factories.cpp
 * Purpose: Implements tensor factory helpers.
 */

#include "bt/tensor.h"

#include <cstdint>
#include <vector>

/*
 * Namespace: bt
 * Purpose: Public BareTensor C++ API surface.
 */
namespace bt {

/*
 * Creates a tensor filled with a constant value.
 */
Tensor full(const std::vector<int64_t> &shape, const float fill_value,
            const bool requires_grad) {
  Tensor tensor(shape);
  tensor.storage->fill(fill_value);
  if (requires_grad) {
    tensor.set_requires_grad(true);
  }
  return tensor;
}

/*
 * Creates a tensor filled with zeros.
 */
Tensor zeros(const std::vector<int64_t> &shape, const bool requires_grad) {
  return full(shape, 0.0f, requires_grad);
}

/*
 * Creates a tensor filled with ones.
 */
Tensor ones(const std::vector<int64_t> &shape, const bool requires_grad) {
  return full(shape, 1.0f, requires_grad);
}

} /* namespace bt */
