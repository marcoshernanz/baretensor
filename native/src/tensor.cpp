#include "bt/tensor.h"

#include <array>
#include <stdexcept>
#include <vector>

Tensor Tensor::full(const std::vector<int> size, float fill_value) {
  if (size.empty()) {
    throw std::runtime_error("Sizes array must not be empty");
  }

  int total_size = 1;
  for (int n : size) {
    if (n <= 0) {
      throw std::runtime_error("Size must be positive");
    }
    total_size *= n;
  }
  data.assign(total_size, fill_value);
  shape = size;
}

Tensor Tensor::zeros(const std::vector<int> size) { return full(size, 0.0f); }

Tensor Tensor::ones(const std::vector<int> size) { return full(size, 1.0f); }

// class Tensor {
//  private:
//   std::vector<float> data;

//  public:
//   std::vector<int> shape;

//   Tensor(const std::vector<int> sizes) {
//     if (sizes.empty()) {
//       throw std::runtime_error("Sizes array must not be empty");
//     }

//     int totalSize = 1;
//     for (int size : sizes) {
//       if (size <= 0) {
//         throw std::runtime_error("Size must be positive");
//       }
//       totalSize *= size;
//     }
//     data.assign(totalSize, 0.0f);
//     shape = sizes;
//   }

//   Tensor operator+(const Tensor& t) {
//     if (shape != t.shape) {
//       throw std::runtime_error("Tensors must have the same shape");
//     }
//   }

//   Tensor operator+(const float n) {
//     std::vector<int> vector = {1};
//     Tensor newTensor = Tensor(vector);
//     newTensor[0] = n;
//     return *this + newTensor;
//   }

//   friend operator+(const float n, const Tensor& t) { return t + n; }
// };
