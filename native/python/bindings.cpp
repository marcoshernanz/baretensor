#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "bt/tensor.h"

namespace nb = nanobind;

NB_MODULE(_C, m) {
  m.doc() = "BareTensor native extension (bootstrap)";

  nb::class_<Tensor>(m, "Tensor")
      .def(nb::init<const std::vector<int>&>())
      .def_ro("shape", &Tensor::shape);
}
