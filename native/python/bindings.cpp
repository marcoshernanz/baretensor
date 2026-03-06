/*
 * File: native/python/bindings.cpp
 * Purpose: Defines Python bindings for the BareTensor native C++ backend.
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstring>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "bt/detail/format.h"
#include "bt/tensor.h"

namespace nb = nanobind;

/*
 * Namespace: (anonymous)
 * Purpose: Private implementation details local to this translation unit.
 */
namespace {

enum class IndexKind { kInt, kSlice };

struct SliceSpec {
  std::optional<int64_t> start;
  std::optional<int64_t> stop;
  std::optional<int64_t> step;
};

struct IndexToken {
  IndexKind kind = IndexKind::kSlice;
  int64_t int_index = 0;
  SliceSpec slice{};
};

[[nodiscard]] IndexToken make_full_slice_token() {
  return IndexToken{
      .kind = IndexKind::kSlice,
      .int_index = 0,
      .slice = SliceSpec{},
  };
}

[[nodiscard]] int64_t cast_index_int(const nb::object &value,
                                     const char *context) {
  if (nb::isinstance<nb::bool_>(value)) {
    throw nb::type_error(
        (std::string(context) +
         " does not support boolean indices. Use int indices instead.")
            .c_str());
  }

  try {
    return nb::cast<int64_t>(value);
  } catch (const nb::cast_error &) {
    throw nb::type_error(
        (std::string(context) + " expected integer-valued index components.")
            .c_str());
  }
}

[[nodiscard]] SliceSpec parse_slice_spec(const nb::slice &slice_obj,
                                         const char *context) {
  const nb::object start_obj = slice_obj.attr("start");
  const nb::object stop_obj = slice_obj.attr("stop");
  const nb::object step_obj = slice_obj.attr("step");

  SliceSpec slice{};
  if (!start_obj.is_none()) {
    slice.start = cast_index_int(start_obj, context);
  }
  if (!stop_obj.is_none()) {
    slice.stop = cast_index_int(stop_obj, context);
  }
  if (!step_obj.is_none()) {
    slice.step = cast_index_int(step_obj, context);
  }
  return slice;
}

[[nodiscard]] IndexToken parse_index_token(const nb::object &index_item,
                                           const char *context) {
  if (index_item.ptr() == Py_Ellipsis) {
    throw nb::type_error((std::string(context) +
                          " does not support ellipsis (...) indexing yet.")
                             .c_str());
  }
  if (nb::isinstance<nb::slice>(index_item)) {
    return IndexToken{
        .kind = IndexKind::kSlice,
        .int_index = 0,
        .slice = parse_slice_spec(nb::cast<nb::slice>(index_item), context),
    };
  }
  if (index_item.is_none()) {
    throw nb::type_error(
        (std::string(context) + " does not support None/newaxis indexing yet.")
            .c_str());
  }

  if (PyIndex_Check(index_item.ptr()) != 0) {
    return IndexToken{
        .kind = IndexKind::kInt,
        .int_index = cast_index_int(index_item, context),
        .slice = SliceSpec{},
    };
  }

  throw nb::type_error(
      (std::string(context) + " only supports int, slice, and tuples thereof.")
          .c_str());
}

[[nodiscard]] std::vector<IndexToken>
normalize_index_tokens(const bt::Tensor &tensor, const nb::object &index_obj,
                       const char *context) {
  std::vector<IndexToken> tokens;
  if (nb::isinstance<nb::tuple>(index_obj)) {
    const nb::tuple tuple_index = nb::cast<nb::tuple>(index_obj);
    tokens.reserve(tuple_index.size());
    for (size_t i = 0; i < tuple_index.size(); ++i) {
      tokens.push_back(
          parse_index_token(nb::borrow<nb::object>(tuple_index[i]), context));
    }
  } else {
    tokens.push_back(parse_index_token(index_obj, context));
  }

  const int64_t rank = tensor.ndim();
  if (static_cast<int64_t>(tokens.size()) > rank) {
    std::ostringstream oss;
    oss << context << " received too many indices for tensor of dimension "
        << rank << ".";
    throw nb::index_error(oss.str().c_str());
  }
  std::vector<IndexToken> normalized = tokens;
  while (static_cast<int64_t>(normalized.size()) < rank) {
    normalized.push_back(make_full_slice_token());
  }
  return normalized;
}

[[nodiscard]] bt::Tensor tensor_getitem(const bt::Tensor &tensor,
                                        const nb::object &index_obj) {
  constexpr const char *kContext = "__getitem__()";
  const std::vector<IndexToken> normalized_tokens =
      normalize_index_tokens(tensor, index_obj, kContext);

  bt::Tensor out = tensor;
  int64_t current_dim = 0;
  for (const IndexToken &token : normalized_tokens) {
    if (token.kind == IndexKind::kInt) {
      out = out.select(current_dim, token.int_index);
      continue;
    }

    const bool is_full_slice =
        !token.slice.start.has_value() && !token.slice.stop.has_value() &&
        (!token.slice.step.has_value() || token.slice.step.value() == 1);
    if (!is_full_slice) {
      const int64_t dim_size = out.shape[static_cast<size_t>(current_dim)];
      const int64_t start = token.slice.start.value_or(0);
      const int64_t stop = token.slice.stop.value_or(dim_size);
      const int64_t step = token.slice.step.value_or(1);
      out = out.slice(current_dim, start, stop, step);
    }
    ++current_dim;
  }

  return out;
}

/*
 * Returns the single scalar value from a tensor with exactly one element.
 */
[[nodiscard]] float tensor_item(const bt::Tensor &tensor) {
  const int64_t count = tensor.numel();
  if (count != 1) {
    std::ostringstream oss;
    oss << "item() can only be called on tensors with exactly one element, "
        << "but got shape " << bt::detail::shape_to_string(tensor.shape)
        << " (" << count << " elements).";
    throw nb::value_error(oss.str().c_str());
  }

  return *tensor.data_ptr();
}

/*
 * Copies contiguous tensor data into a std::vector for Python conversion
 * helpers.
 */
[[nodiscard]] std::vector<float> tensor_to_vector(const bt::Tensor &t) {
  std::vector<float> out(static_cast<size_t>(t.numel()));
  if (!out.empty()) {
    std::memcpy(out.data(), t.data_ptr(), out.size() * sizeof(float));
  }
  return out;
}

/*
 * Builds a full reduction-dimension list [0, 1, ..., ndim - 1].
 */
[[nodiscard]] std::vector<int64_t> make_all_dims(const bt::Tensor &tensor) {
  std::vector<int64_t> dims(static_cast<size_t>(tensor.ndim()), 0);
  for (size_t i = 0; i < dims.size(); ++i) {
    dims[i] = static_cast<int64_t>(i);
  }
  return dims;
}

/*
 * Dispatches a reduction call where dim can be None, int, or sequence[int].
 */
template <typename SingleDimReducer, typename MultiDimReducer>
[[nodiscard]] bt::Tensor
dispatch_reduction_call(const bt::Tensor &tensor, nb::object dim,
                        const bool keepdim, const char *operation_name,
                        const SingleDimReducer &single_dim_reducer,
                        const MultiDimReducer &multi_dim_reducer) {
  if (dim.is_none()) {
    return multi_dim_reducer(make_all_dims(tensor), keepdim);
  }
  if (nb::isinstance<nb::int_>(dim)) {
    return single_dim_reducer(nb::cast<int64_t>(dim), keepdim);
  }

  try {
    return multi_dim_reducer(nb::cast<std::vector<int64_t>>(dim), keepdim);
  } catch (const nb::cast_error &) {
    const std::string message =
        std::string(operation_name) +
        "() expected 'dim' to be an int, a sequence of ints, or None.";
    throw nb::type_error(message.c_str());
  }
}

/*
 * Sets requires_grad on a tensor and returns the same tensor handle.
 */
bt::Tensor &set_requires_grad(bt::Tensor &tensor, const bool requires_grad) {
  return tensor.set_requires_grad(requires_grad);
}

} // namespace

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
      .def("ndim", &bt::Tensor::ndim)
      .def("numel", &bt::Tensor::numel)
      .def_prop_rw(
          "requires_grad", &bt::Tensor::requires_grad,
          [](bt::Tensor &tensor, const bool requires_grad) {
            set_requires_grad(tensor, requires_grad);
          },
          "Whether this tensor tracks gradients in autograd.")
      .def("set_requires_grad", &set_requires_grad, nb::arg("requires_grad"),
           nb::rv_policy::reference_internal)
      .def_prop_ro("is_leaf", &bt::Tensor::is_leaf)
      .def_prop_ro("grad", &bt::Tensor::grad)
      .def("zero_grad", &bt::Tensor::zero_grad)
      .def("detach", &bt::Tensor::detach)
      .def("backward", &bt::Tensor::backward,
           nb::arg("gradient") = std::nullopt)
      .def("is_contiguous", &bt::Tensor::is_contiguous)
      .def("contiguous", &bt::Tensor::contiguous)
      .def("view", &bt::Tensor::view, nb::arg("shape"))
      .def("reshape", &bt::Tensor::reshape, nb::arg("shape"))
      .def("permute", &bt::Tensor::permute, nb::arg("dims"))
      .def("transpose", &bt::Tensor::transpose, nb::arg("dim0"),
           nb::arg("dim1"))
      .def("__getitem__", &tensor_getitem, nb::arg("index").none())
      .def("item", &tensor_item)
      .def_prop_ro("T", &bt::Tensor::T)
      .def_prop_ro("mT", &bt::Tensor::mT)
      .def("matmul", &bt::Tensor::matmul, nb::arg("tensor2"))
      .def("__matmul__", [](const bt::Tensor &lhs,
                            const bt::Tensor &rhs) { return lhs.matmul(rhs); })
      .def("__neg__",
           static_cast<bt::Tensor (bt::Tensor::*)() const>(&bt::Tensor::operator-))
      .def("exp", &bt::Tensor::exp)
      .def("log", &bt::Tensor::log)
      .def("tanh", &bt::Tensor::tanh)
      .def("softmax", &bt::Tensor::softmax, nb::arg("dim"))
      .def("log_softmax", &bt::Tensor::log_softmax, nb::arg("dim"))
      .def(
          "sum",
          [](const bt::Tensor &tensor, nb::object dim, const bool keepdim) {
            return dispatch_reduction_call(
                tensor, dim, keepdim, "sum",
                [&tensor](const int64_t one_dim, const bool keepdim_inner) {
                  return tensor.sum(one_dim, keepdim_inner);
                },
                [&tensor](const std::vector<int64_t> &many_dims,
                          const bool keepdim_inner) {
                  return tensor.sum(many_dims, keepdim_inner);
                });
          },
          nb::arg("dim") = nb::none(), nb::arg("keepdim") = false)
      .def(
          "mean",
          [](const bt::Tensor &tensor, nb::object dim, const bool keepdim) {
            return dispatch_reduction_call(
                tensor, dim, keepdim, "mean",
                [&tensor](const int64_t one_dim, const bool keepdim_inner) {
                  return tensor.mean(one_dim, keepdim_inner);
                },
                [&tensor](const std::vector<int64_t> &many_dims,
                          const bool keepdim_inner) {
                  return tensor.mean(many_dims, keepdim_inner);
                });
          },
          nb::arg("dim") = nb::none(), nb::arg("keepdim") = false)
      .def(
          "max",
          [](const bt::Tensor &tensor, nb::object dim, const bool keepdim) {
            return dispatch_reduction_call(
                tensor, dim, keepdim, "max",
                [&tensor](const int64_t one_dim, const bool keepdim_inner) {
                  return tensor.max(one_dim, keepdim_inner);
                },
                [&tensor](const std::vector<int64_t> &many_dims,
                          const bool keepdim_inner) {
                  return tensor.max(many_dims, keepdim_inner);
                });
          },
          nb::arg("dim") = nb::none(), nb::arg("keepdim") = false)
      .def("numpy",
           [numpy](const bt::Tensor &t) {
             const bt::Tensor contiguous = t.contiguous();
             const std::vector<float> values = tensor_to_vector(contiguous);
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

  m.def("full", &bt::full, nb::arg("shape"), nb::arg("fill_value"),
        nb::arg("requires_grad") = false);
  m.def("zeros", &bt::zeros, nb::arg("shape"),
        nb::arg("requires_grad") = false);
  m.def("ones", &bt::ones, nb::arg("shape"), nb::arg("requires_grad") = false);
  m.def("cross_entropy", &bt::cross_entropy, nb::arg("input"),
        nb::arg("target"), nb::arg("ignore_index") = -100,
        nb::arg("reduction") = "mean");
  m.def("layer_norm", &bt::layer_norm, nb::arg("input"),
        nb::arg("normalized_shape"), nb::arg("weight") = std::nullopt,
        nb::arg("bias") = std::nullopt, nb::arg("eps") = 1e-5f);
  m.def("embedding", &bt::embedding, nb::arg("input"), nb::arg("weight"));

  m.def(
      "tensor",
      [](const NdArrayF32 &a, const bool requires_grad) {
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
        if (requires_grad) {
          t.set_requires_grad(true);
        }
        return t;
      },
      nb::arg("array"), nb::arg("requires_grad") = false);
}
