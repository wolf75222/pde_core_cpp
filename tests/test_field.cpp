// Field1D / Field2D templated on Cell.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <pde_core/mesh/domain1d.hpp>
#include <pde_core/mesh/domain2d.hpp>
#include <pde_core/mesh/field1d.hpp>
#include <pde_core/mesh/field2d.hpp>

#include <array>

using Catch::Matchers::WithinAbs;

using pde_core::Domain1D;
using pde_core::Domain2D;
using pde_core::Field1D;
using pde_core::Field2D;
using pde_core::Real;

TEST_CASE("Field1D<Real> default-initialised to zero", "[field][1d][scalar]") {
  Field1D<Real> f(8, 2);
  REQUIRE(f.nx() == 8);
  REQUIRE(f.ghost() == 2);
  REQUIRE(f.total_size() == 12);
  for (std::ptrdiff_t i = -2; i < 10; ++i) {
    REQUIRE(f(i) == Real{0});
  }
}

TEST_CASE("Field1D<Real> ghost-cell access is signed-index", "[field][1d][ghost]") {
  Field1D<Real> f(4, 2);
  f(-1) = Real{1.5};
  f(4)  = Real{2.5};
  // Underlying buffer : data_[ghost + i] is cell i.
  REQUIRE(f.data()[1] == Real{1.5});  // i = -1 -> buffer index 2 - 1 = 1
  REQUIRE(f.data()[6] == Real{2.5});  // i = 4  -> buffer index 2 + 4 = 6
}

TEST_CASE("Field2D<Real> default-initialised to zero", "[field][2d][scalar]") {
  Field2D<Real> f(4, 6, 2);
  REQUIRE(f.nx() == 4);
  REQUIRE(f.ny() == 6);
  REQUIRE(f.ghost() == 2);
  for (std::ptrdiff_t j = -2; j < 8; ++j)
    for (std::ptrdiff_t i = -2; i < 6; ++i)
      REQUIRE(f(i, j) == Real{0});
}

// A trivial std::array-like Cell with ::Zero() to exercise the CellTraits
// specialisation that detects Eigen-style static Zero().
struct FakeVecCell {
  std::array<Real, 3> a{};
  static FakeVecCell Zero() { return FakeVecCell{{Real{0}, Real{0}, Real{0}}}; }
};

TEST_CASE("Field2D<FakeVecCell> picks up CellTraits<Eigen-like>::Zero()",
          "[field][2d][traits]") {
  Field2D<FakeVecCell> f(2, 2, 1);
  for (std::ptrdiff_t j = -1; j < 3; ++j)
    for (std::ptrdiff_t i = -1; i < 3; ++i) {
      REQUIRE(f(i, j).a[0] == Real{0});
      REQUIRE(f(i, j).a[1] == Real{0});
      REQUIRE(f(i, j).a[2] == Real{0});
    }
}

TEST_CASE("Domain1D cell coordinates", "[domain][1d]") {
  Domain1D dom{0.0, 1.0, 10};
  REQUIRE_THAT(dom.dx(), WithinAbs(0.1, 1e-15));
  REQUIRE_THAT(dom.x_cell(0), WithinAbs(0.05, 1e-15));
  REQUIRE_THAT(dom.x_cell(9), WithinAbs(0.95, 1e-15));
}

TEST_CASE("Domain2D cell coordinates", "[domain][2d]") {
  Domain2D dom{0.0, 2.0, 0.0, 4.0, 4, 8};
  REQUIRE(dom.dx() == Real{0.5});
  REQUIRE(dom.dy() == Real{0.5});
  REQUIRE(dom.x_cell(0) == Real{0.25});
  REQUIRE(dom.y_cell(7) == Real{3.75});
}
