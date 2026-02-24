#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/vector.h>

#include <cstring>

#include "bt/tensor.h"

namespace nb = nanobind;

NB_MODULE(_C, m) {
  m.doc() = "BareTensor native extension (bootstrap)";

  using NdArrayF32 =
      nb::ndarray<nb::numpy, const float, nb::c_contig, nb::device::cpu>;

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

  m.def("tensor", [](const NdArrayF32& a) {
    std::vector<int64_t> shape(a.ndim());
    for (size_t i = 0; i < a.ndim(); ++i) {
      shape.push_back(static_cast<int64_t>(a.shape(i)));
    }

    bt::Tensor t(shape);
    if (a.nbytes() != 0) {
      std::memcpy(t.data_ptr(), a.data(), a.nbytes());
    }
    return t;
  });
}
