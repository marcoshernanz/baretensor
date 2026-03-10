/*
 * File: native/include/bt/dtype.h
 * Purpose: Declares runtime dtype metadata and dispatch helpers.
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace bt {

/*
 * Enum: ScalarKind
 * Purpose: Categorizes supported runtime scalar types.
 */
enum class ScalarKind { kFloatingPoint, kIntegral };

/*
 * Enum: ScalarType
 * Purpose: Identifies the concrete scalar type stored by a tensor.
 */
enum class ScalarType { kFloat32, kInt64 };

/*
 * Returns the canonical user-facing name for a scalar type.
 */
[[nodiscard]] constexpr const char *scalar_type_name(const ScalarType type) noexcept {
  switch (type) {
  case ScalarType::kFloat32:
    return "float32";
  case ScalarType::kInt64:
    return "int64";
  }
  return "<unknown>";
}

/*
 * Returns the element size in bytes for a scalar type.
 */
[[nodiscard]] constexpr size_t scalar_type_itemsize(const ScalarType type) noexcept {
  switch (type) {
  case ScalarType::kFloat32:
    return sizeof(float);
  case ScalarType::kInt64:
    return sizeof(int64_t);
  }
  return 0;
}

/*
 * Returns the high-level category of a scalar type.
 */
[[nodiscard]] constexpr ScalarKind scalar_type_kind(const ScalarType type) noexcept {
  switch (type) {
  case ScalarType::kFloat32:
    return ScalarKind::kFloatingPoint;
  case ScalarType::kInt64:
    return ScalarKind::kIntegral;
  }
  return ScalarKind::kIntegral;
}

/*
 * Returns whether a scalar type is floating-point.
 */
[[nodiscard]] constexpr bool is_floating_point(const ScalarType type) noexcept {
  return scalar_type_kind(type) == ScalarKind::kFloatingPoint;
}

/*
 * Returns whether a scalar type is integral.
 */
[[nodiscard]] constexpr bool is_integral(const ScalarType type) noexcept {
  return scalar_type_kind(type) == ScalarKind::kIntegral;
}

template <typename T> struct ScalarTypeTraits;

template <> struct ScalarTypeTraits<float> {
  static constexpr ScalarType value = ScalarType::kFloat32;
};

template <> struct ScalarTypeTraits<int64_t> {
  static constexpr ScalarType value = ScalarType::kInt64;
};

/*
 * Returns the runtime scalar type corresponding to a C++ scalar type.
 */
template <typename T> [[nodiscard]] constexpr ScalarType scalar_type_of() noexcept {
  using Scalar = std::remove_cv_t<T>;
  return ScalarTypeTraits<Scalar>::value;
}

/*
 * Invokes a templated callable with the C++ scalar corresponding to type.
 */
template <typename Fn> decltype(auto) visit_dtype(const ScalarType type, Fn &&fn) {
  switch (type) {
  case ScalarType::kFloat32:
    return std::forward<Fn>(fn).template operator()<float>();
  case ScalarType::kInt64:
    return std::forward<Fn>(fn).template operator()<int64_t>();
  }
  throw std::invalid_argument("Unsupported dtype dispatch request.");
}

/*
 * Invokes a templated callable only for floating-point dtypes.
 */
template <typename Fn>
decltype(auto) visit_floating_dtype(const ScalarType type, const std::string_view context,
                                    Fn &&fn) {
  switch (type) {
  case ScalarType::kFloat32:
    return std::forward<Fn>(fn).template operator()<float>();
  case ScalarType::kInt64:
    throw std::invalid_argument(std::string(context) +
                                " only supports floating-point tensors, but got dtype " +
                                scalar_type_name(type) + ".");
  }
  throw std::invalid_argument("Unsupported floating-point dtype dispatch request.");
}

/*
 * Converts a floating-point scalar to int64 with fail-loud validation.
 */
[[nodiscard]] inline int64_t checked_int64_from_double(const double value,
                                                       const std::string_view context) {
  if (!std::isfinite(value)) {
    throw std::invalid_argument(std::string(context) +
                                " expected a finite integer-valued scalar.");
  }

  const double truncated = std::trunc(value);
  if (truncated != value) {
    throw std::invalid_argument(std::string(context) +
                                " expected an integer-valued scalar.");
  }

  const double int64_min =
      static_cast<double>(std::numeric_limits<int64_t>::lowest());
  const double int64_max = static_cast<double>(std::numeric_limits<int64_t>::max());
  if (truncated < int64_min || truncated > int64_max) {
    throw std::invalid_argument(std::string(context) +
                                " expected a scalar in the int64 range.");
  }

  return static_cast<int64_t>(truncated);
}

} // namespace bt
