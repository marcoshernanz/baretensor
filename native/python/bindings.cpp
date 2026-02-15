#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "bt/tensor.h"
#include "vector"

namespace nb = nanobind;

class Tensor {
 public:
  int a;
  Tensor(int a) { this->a = a; }
};

NB_MODULE(_C, m) {
  m.doc() = "BareTensor";

  nb::class_<Tensor>(m, "Tensor").def(nb::init<int>());
  //   nb::class_<Tensor>(m, "Tensor").def(nb::init<std::vector<int>>());
  //   .def(nb::init<const std::string&>())
  //   .def("bark", &Dog::bark)
  //   .def_rw("name", &Dog::name);
}
