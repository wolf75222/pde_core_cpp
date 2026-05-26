cmake_minimum_required(VERSION 3.20)
project(pde_core_cpp
  VERSION 0.1.0
  DESCRIPTION "Shared C++20 core for advection_cpp + euler_cpp : AMR primitives, mesh, integrators."
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(PDE_CORE_BUILD_TESTS "Build pde_core_cpp tests" OFF)
option(PDE_CORE_USE_OPENMP "Enable OpenMP pragmas in pde_core" OFF)

# Header-only INTERFACE library.
add_library(pde_core INTERFACE)
add_library(pde_core::pde_core ALIAS pde_core)

target_include_directories(pde_core INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_compile_features(pde_core INTERFACE cxx_std_20)

# OpenMP : optional. Consumers can also enable it themselves and pass the
# PDE_CORE_HAS_OPENMP define ; we expose it as a transitive flag here so a
# single FetchContent + option(... ON) is enough.
if(PDE_CORE_USE_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(pde_core INTERFACE OpenMP::OpenMP_CXX)
  target_compile_definitions(pde_core INTERFACE PDE_CORE_HAS_OPENMP)
endif()

if(PDE_CORE_BUILD_TESTS)
  enable_testing()
  include(FetchContent)
  FetchContent_Declare(
    catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.2)
  FetchContent_MakeAvailable(catch2)
  add_subdirectory(tests)
endif()
