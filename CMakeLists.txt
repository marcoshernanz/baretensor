cmake_minimum_required(VERSION 3.28)

project(baretensor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(Python 3.13 COMPONENTS Interpreter Development.Module REQUIRED)

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT
)

find_package(nanobind CONFIG REQUIRED)

set(BT_DETAIL_SOURCES
  native/src/detail/broadcast.cpp
  native/src/detail/dims.cpp
  native/src/detail/format.cpp
  native/src/detail/shape.cpp
)

set(BT_CORE_SOURCES
  native/src/ops.cpp
  native/src/storage.cpp
  native/src/tensor.cpp
)

nanobind_add_module(
  _C
  NOMINSIZE
  native/python/bindings.cpp
  ${BT_DETAIL_SOURCES}
  ${BT_CORE_SOURCES}
)

target_include_directories(_C PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/native/include"
)

# Add include dirs here as you grow (native/include, native/src, etc.).

# Make the module importable without installation:
#   PYTHONPATH=src python -c "import bt; ..."
set_target_properties(_C PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/bt"
)

# Generate typing stubs at build time.
# This runs nanobind's stub generator after the module is built, and writes
# the result next to the extension so that Pyright/VS Code can pick it up.
nanobind_add_stub(
  bt_stub
  MODULE bt._C
  OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/src/bt/_C.pyi"
  MARKER_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/bt/py.typed"
  PYTHON_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src"
  DEPENDS _C
)
