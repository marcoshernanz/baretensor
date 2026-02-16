#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "bt/tensor.h"

namespace nb = nanobind;

NB_MODULE(_C, m) {
  m.doc() = "BareTensor native extension (bootstrap)";

  nb::class_<Tensor>(m, "Tensor").def_ro("shape", &Tensor::shape);
  m.def("full", &full);
  m.def("zeros", &zeros);
  m.def("ones", &ones);
}
