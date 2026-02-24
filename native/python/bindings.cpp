#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "bt/tensor.h"

namespace nb = nanobind;

namespace {

[[nodiscard]] std::vector<float> tensor_to_vector(const bt::Tensor& t) {
  std::vector<float> out(static_cast<size_t>(t.numel()));
  if (!out.empty()) {
    std::memcpy(out.data(), t.data_ptr(), out.size() * sizeof(float));
  }
  return out;
}

[[nodiscard]] std::string shape_to_string(const std::vector<int64_t>& shape) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) oss << ", ";
    oss << shape[i];
  }
  oss << "]";
  return oss.str();
}

}  // namespace

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

  m.def("tensor", [](const NdArrayF32& a) {
    std::vector<int64_t> shape;
    shape.reserve(a.ndim());
    for (size_t i = 0; i < a.ndim(); ++i) {
      shape.push_back(static_cast<int64_t>(a.shape(i)));
    }

    bt::Tensor t(shape);
    const size_t expected_nbytes = static_cast<size_t>(t.numel()) * sizeof(float);
    if (a.nbytes() != expected_nbytes) {
      std::ostringstream oss;
      oss << "Failed to copy NumPy array into Tensor(shape="
          << shape_to_string(shape) << "): expected " << expected_nbytes
          << " bytes but got " << a.nbytes() << ".";
      throw std::runtime_error(oss.str());
    }
    if (a.nbytes() != 0) {
      std::memcpy(t.data_ptr(), a.data(), a.nbytes());
    }
    return t;
  }, nb::arg("array"));
}
