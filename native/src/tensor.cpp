#include <vector>

class Tensor {
 public:
  int a;
  Tensor(int a) { this->a = a; }
};

// class Tensor {
//  private:
//   std::vector<float> data;

//  public:
//   std::vector<int> shape;

//   Tensor(std::vector<int> sizes) {
//     if (sizes.size() < 1) throw "Sizes array must not be empty";

//     int totalSize = 1;
//     for (int& size : sizes) {
//       if (size <= 0) throw "Size must be positive";
//       totalSize *= size;
//     }

//     this->data.resize(totalSize);
//     this->shape = sizes;
//   }
// };