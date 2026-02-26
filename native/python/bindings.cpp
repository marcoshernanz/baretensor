/*
 * File: native/python/bindings.cpp
 * Purpose: Defines Python bindings for the BareTensor native C++ backend.
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "bt/detail/format.h"
#include "bt/tensor.h"

namespace nb = nanobind;

/*
 * Namespace: (anonymous)
 * Purpose: Private implementation details local to this translation unit.
 */
namespace {

/*
 * Copies tensor data into a std::vector for Python conversion helpers.
 */
[[nodiscard]] std::vector<float> tensor_to_vector(const bt::Tensor& t) {
  std::vector<float> out(static_cast<size_t>(t.numel()));
  if (!out.empty()) {
    std::memcpy(out.data(), t.data_ptr(), out.size() * sizeof(float));
  }
  return out;
}

}  // namespace

/*
 * Defines module bt._C and binds Tensor and factory functions to Python.
 */
NB_MODULE(_C, m) {
  m.doc() = "BareTensor native extension (bootstrap)";
  nb::module_ numpy = nb::module_::import_("numpy");

  using NdArrayF32 =
      nb::ndarray<nb::numpy, const float, nb::c_contig, nb::device::cpu>;

  nb::class_<bt::Tensor>(m, "Tensor")
      .def_ro("shape", &bt::Tensor::shape)
      .def_ro("strides", &bt::Tensor::strides)
      .def("dim", &bt::Tensor::dim)
      .def("numel", &bt::Tensor::numel)
      .def("is_contiguous", &bt::Tensor::is_contiguous)
      .def("contiguous", &bt::Tensor::contiguous)
      .def("view", &bt::Tensor::view, nb::arg("shape"))
      .def("reshape", &bt::Tensor::reshape, nb::arg("shape"))
      .def("numpy",
           [numpy](const bt::Tensor& t) {
             const std::vector<float> values = tensor_to_vector(t);
             nb::object array = numpy.attr("array")(
                 nb::cast(values), nb::arg("dtype") = numpy.attr("float32"));
             return array.attr("reshape")(nb::cast(t.shape));
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

  m.def(
      "tensor",
      [](const NdArrayF32& a) {
        std::vector<int64_t> shape;
        shape.reserve(a.ndim());
        for (size_t i = 0; i < a.ndim(); ++i) {
          shape.push_back(static_cast<int64_t>(a.shape(i)));
        }

        bt::Tensor t(shape);
        const size_t expected_nbytes =
            static_cast<size_t>(t.numel()) * sizeof(float);
        if (a.nbytes() != expected_nbytes) {
          std::ostringstream oss;
          oss << "Failed to copy NumPy array into Tensor(shape="
              << bt::detail::shape_to_string(shape) << "): expected "
              << expected_nbytes << " bytes but got " << a.nbytes() << ".";
          throw std::runtime_error(oss.str());
        }
        if (a.nbytes() != 0) {
          std::memcpy(t.data_ptr(), a.data(), a.nbytes());
        }
        return t;
      },
      nb::arg("array"));
}
