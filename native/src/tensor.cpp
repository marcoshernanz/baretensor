#include <array>
#include <vector>

class Tensor {
 public:
  int a;
  Tensor(int a) { this->a = a; }
};

class Tensor {
 private:
  std::vector<float> data;

 public:
  std::vector<int> shape;

  Tensor(const std::vector<int> sizes) {
    if (sizes.empty()) {
      throw std::runtime_error("Sizes array must not be empty");
    }

    int totalSize = 1;
    for (int size : sizes) {
      if (size <= 0) {
        throw std::runtime_error("Size must be positive");
      }
      totalSize *= size;
    }
    data.assign(totalSize, 0.0f);
    shape = sizes;
  }

  Tensor operator+(const Tensor& t) {
    if (shape != t.shape) {
      throw std::runtime_error("Tensors must have the same shape");
    }
  }

  Tensor operator+(const float n) {
    std::vector<int> vector = {1};
    Tensor newTensor = Tensor(vector);
    newTensor[0] = n;
    return *this + newTensor;
  }

  friend operator+(float n, const Tensor& t) { return t + n; }
};
