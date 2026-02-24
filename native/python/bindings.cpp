#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>

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
      .def("tolist",
           [](const bt::Tensor& t) {
             std::vector<float> out(static_cast<size_t>(t.numel()));
             if (!out.empty()) {
               std::memcpy(out.data(), t.data_ptr(), out.size() * sizeof(float));
             }
             return out;
           })
      .def(nb::self + nb::self)
      .def(nb::self + float())
      .def(nb::self - nb::self)
      .def(nb::self - float())
      .def(nb::self * nb::self)
      .def(nb::self * float())
      .def(nb::self / nb::self)
      .def(nb::self / float());

  m.def("full", &bt::full, nb::arg("shape"), nb::arg("fill_value"));
  m.def("zeros", &bt::zeros, nb::arg("shape"));
  m.def("ones", &bt::ones, nb::arg("shape"));

  m.def("tensor", [](const NdArrayF32& a) {
    std::vector<int64_t> shape;
    shape.reserve(a.ndim());
    for (size_t i = 0; i < a.ndim(); ++i) {
      shape.push_back(static_cast<int64_t>(a.shape(i)));
    }

    bt::Tensor t(shape);
    const size_t expected_nbytes = static_cast<size_t>(t.numel()) * sizeof(float);
    if (a.nbytes() != expected_nbytes) {
      throw std::runtime_error("Unexpected NumPy array byte size for tensor copy");
    }
    if (a.nbytes() != 0) {
      std::memcpy(t.data_ptr(), a.data(), a.nbytes());
    }
    return t;
  }, nb::arg("array"));
}
