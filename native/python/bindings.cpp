#include <bt/tensor.h>
#include <nanobind/nanobind.h>

namespace nb = nanobind;

NB_MODULE(_C, m) {
  m.doc() = "BareTensor native extension (bootstrap)";

  nb::class_<Tensor>(m, "Tensor").def(nb::init<const int>());
}
