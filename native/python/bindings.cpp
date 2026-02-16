#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "bt/tensor.h"

namespace nb = nanobind;

NB_MODULE(_C, m) {
  m.doc() = "BareTensor native extension (bootstrap)";

  nb::class_<bt::Tensor>(m, "Tensor")
      .def_ro("shape", &bt::Tensor::shape)
      .def("__repr__", &bt::Tensor::__repr__)
      .def("stride", nb::overload_cast<>(&bt::Tensor::stride, nb::const_))
      .def("stride", nb::overload_cast<int>(&bt::Tensor::stride, nb::const_));

  m.def("full", &bt::full);
  m.def("zeros", &bt::zeros);
  m.def("ones", &bt::ones);
}
