#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "bt/tensor.h"

namespace nb = nanobind;

NB_MODULE(_C, m) {
  m.doc() = "BareTensor native extension (bootstrap)";

  nb::class_<bt::Tensor>(m, "Tensor")
      .def_ro("shape", &bt::Tensor::shape)
      .def("stride", [](const bt::Tensor& t) { return t.stride(); })
      .def("stride",
           [](const bt::Tensor& t, int dim) { return t.stride(dim); });

  m.def("full", &bt::full);
  m.def("zeros", &bt::zeros);
  m.def("ones", &bt::ones);
}
