#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/vector.h>

#include "bt/tensor.h"

namespace nb = nanobind;

NB_MODULE(_C, m) {
  m.doc() = "BareTensor native extension (bootstrap)";

  nb::class_<bt::Tensor>(m, "Tensor")
      .def_ro("shape", &bt::Tensor::shape)
      .def_ro("strides", &bt::Tensor::strides)
      .def("dim", &bt::Tensor::dim)
      .def("numel", &bt::Tensor::numel)
      .def("is_contiguous", &bt::Tensor::is_contiguous)
      .def(nb::self + nb::self)
      .def(nb::self + float())
      .def(nb::self - nb::self)
      .def(nb::self - float())
      .def(nb::self * nb::self)
      .def(nb::self * float())
      .def(nb::self / nb::self)
      .def(nb::self / float());

  m.def("full", &bt::full);
  m.def("zeros", &bt::zeros);
  m.def("ones", &bt::ones);
}
