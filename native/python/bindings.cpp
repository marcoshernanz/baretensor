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

// Tensor tensor(const nb::ndarray<float>& array) {
//   std::vector<int64_t> shape(array.ndim());
//   std::vector<int64_t> strides(array.ndim());
//   for (int i = 0; i < array.ndim(); i++) {
//     shape[i] = array.shape(i);
//     strides[i] = array.stride(i);
//   }

//   Tensor tensor(shape);
//   tensor.strides = strides;
//   auto x = array.data();
//   tensor.storage->data = static_cast<float*>(array.data());
// }
