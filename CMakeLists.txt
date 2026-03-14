cmake_minimum_required(VERSION 3.28)

project(baretensor LANGUAGES CXX)

include(CheckIPOSupported)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang|GNU")
  string(APPEND CMAKE_CXX_FLAGS_RELEASE " -ffp-contract=fast")
elseif (MSVC)
  string(APPEND CMAKE_CXX_FLAGS_RELEASE " /fp:fast")
endif()

check_ipo_supported(RESULT BT_IPO_SUPPORTED OUTPUT BT_IPO_ERROR)
if (BT_IPO_SUPPORTED)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

find_package(Python 3.13 COMPONENTS Interpreter Development.Module REQUIRED)

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT
)

find_package(nanobind CONFIG REQUIRED)

# Automatically pick up all C++ implementation files under native/src.
# CONFIGURE_DEPENDS asks CMake to re-run configure when files are added/removed.
file(GLOB_RECURSE BT_NATIVE_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/native/src/*.cpp"
)
list(SORT BT_NATIVE_SOURCES)

nanobind_add_module(
  _C
  NOMINSIZE
  native/python/bindings.cpp
  ${BT_NATIVE_SOURCES}
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
