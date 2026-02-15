#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <string>

namespace nb = nanobind;
using namespace nb::literals;

static int add(int a, int b = 1) { return a + b; }

struct Dog {
    std::string name;

    Dog() = default;
    explicit Dog(const std::string &name) : name(name) {}

    std::string bark() const { return name + ": woof!"; }
};

NB_MODULE(_C, m) {
    m.doc() = "BareTensor native extension (bootstrap)";

    m.def("add", &add, "a"_a, "b"_a = 1,
          "Add two integers (default b=1).\n\n"
          "This exists only to validate the nanobind toolchain.");

    nb::class_<Dog>(m, "Dog")
        .def(nb::init<>())
        .def(nb::init<const std::string &>())
        .def("bark", &Dog::bark)
        .def_rw("name", &Dog::name);
}
