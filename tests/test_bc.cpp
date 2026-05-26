// Periodic + Outflow BCs in 1D and 2D, on scalar and 4-vector fields.

#include <catch2/catch_test_macros.hpp>

#include <pde_core/bc/outflow.hpp>
#include <pde_core/bc/outflow_2d.hpp>
#include <pde_core/bc/periodic.hpp>
#include <pde_core/bc/periodic_2d.hpp>
#include <pde_core/mesh/field1d.hpp>
#include <pde_core/mesh/field2d.hpp>

#include <array>

using pde_core::Field1D;
using pde_core::Field2D;
using pde_core::OutflowBC1D;
using pde_core::OutflowBC2D;
using pde_core::PeriodicBC1D;
using pde_core::PeriodicBC2D;
using pde_core::Real;

TEST_CASE("PeriodicBC1D<Real> : ghost cells mirror the opposite side",
          "[bc][periodic][1d]") {
  Field1D<Real> U(8, /*ghost=*/2);
  for (std::ptrdiff_t i = 0; i < 8; ++i) U(i) = static_cast<Real>(i + 10);
  PeriodicBC1D{}.apply(U);
  REQUIRE(U(-1) == U(7));
  REQUIRE(U(-2) == U(6));
  REQUIRE(U(8)  == U(0));
  REQUIRE(U(9)  == U(1));
}

TEST_CASE("OutflowBC1D<Real> : ghost cells copy the interior boundary",
          "[bc][outflow][1d]") {
  Field1D<Real> U(8, 2);
  for (std::ptrdiff_t i = 0; i < 8; ++i) U(i) = static_cast<Real>(i + 10);
  OutflowBC1D{}.apply(U);
  REQUIRE(U(-1) == U(0));
  REQUIRE(U(-2) == U(0));
  REQUIRE(U(8)  == U(7));
  REQUIRE(U(9)  == U(7));
}

TEST_CASE("PeriodicBC2D<Real> : edges + corners all match opposite side",
          "[bc][periodic][2d]") {
  Field2D<Real> U(4, 4, 2);
  for (std::ptrdiff_t j = 0; j < 4; ++j)
    for (std::ptrdiff_t i = 0; i < 4; ++i)
      U(i, j) = static_cast<Real>(10 * (j + 1) + i);
  PeriodicBC2D{}.apply(U);
  // Edges
  for (std::ptrdiff_t j = 0; j < 4; ++j) {
    REQUIRE(U(-1, j) == U(3, j));
    REQUIRE(U(4, j)  == U(0, j));
  }
  for (std::ptrdiff_t i = 0; i < 4; ++i) {
    REQUIRE(U(i, -1) == U(i, 3));
    REQUIRE(U(i, 4)  == U(i, 0));
  }
  // Corners (filled by the second pass over the full -g..nx+g span)
  REQUIRE(U(-1, -1) == U(3, 3));
  REQUIRE(U(4, 4)   == U(0, 0));
  REQUIRE(U(-1, 4)  == U(3, 0));
  REQUIRE(U(4, -1)  == U(0, 3));
}

TEST_CASE("OutflowBC2D<Real> : ghost cells copy the closest interior cell",
          "[bc][outflow][2d]") {
  Field2D<Real> U(4, 4, 2);
  for (std::ptrdiff_t j = 0; j < 4; ++j)
    for (std::ptrdiff_t i = 0; i < 4; ++i)
      U(i, j) = static_cast<Real>(10 * (j + 1) + i);
  OutflowBC2D{}.apply(U);
  for (std::ptrdiff_t j = 0; j < 4; ++j) {
    REQUIRE(U(-1, j) == U(0, j));
    REQUIRE(U(4, j)  == U(3, j));
  }
  for (std::ptrdiff_t i = -2; i < 6; ++i) {
    REQUIRE(U(i, -1) == U(i, 0));
    REQUIRE(U(i, 4)  == U(i, 3));
  }
}

// A fake 4-vector cell to exercise the BCs on system fields without
// pulling Eigen into the pde_core test binary.
struct Vec4 {
  std::array<Real, 4> a{};
  static Vec4 Zero() { return Vec4{{Real{0}, Real{0}, Real{0}, Real{0}}}; }
  bool operator==(const Vec4& o) const { return a == o.a; }
};

TEST_CASE("PeriodicBC2D works on a system field (Field2D<Vec4>)",
          "[bc][periodic][2d][system]") {
  Field2D<Vec4> U(4, 4, 2);
  for (std::ptrdiff_t j = 0; j < 4; ++j)
    for (std::ptrdiff_t i = 0; i < 4; ++i)
      U(i, j) = Vec4{{static_cast<Real>(i), static_cast<Real>(j),
                       static_cast<Real>(i + j), static_cast<Real>(i * j)}};
  PeriodicBC2D{}.apply(U);
  REQUIRE(U(-1, 0) == U(3, 0));
  REQUIRE(U(0, -1) == U(0, 3));
  REQUIRE(U(-1, -1) == U(3, 3));
}
