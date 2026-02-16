#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "bt/tensor.h"

namespace nb = nanobind;

NB_MODULE(_C, m) {
  m.doc() = "BareTensor native extension (bootstrap)";

  nb::class_<Tensor>(m, "Tensor")
      .def_ro("shape", &Tensor::shape)
      .def("fill", &Tensor::full)
      .def("fill", &Tensor::zeros)
      .def("fill", &Tensor::ones);
}
