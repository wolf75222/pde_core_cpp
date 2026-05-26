// FluxRegister1D / FluxRegister2D templated on Cell. Verify that the
// Kahan-compensated accumulation is bit-stable, that the scalar fallback
// works for Real, and that the Eigen-style fallback works for a 4-vec.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <pde_core/amr/flux_register1d.hpp>
#include <pde_core/amr/flux_register2d.hpp>

#include <array>

using Catch::Matchers::WithinAbs;
using pde_core::FluxRegister1D;
using pde_core::FluxRegister2D;
using pde_core::Real;

// FakeVec4 : a 4-vec Cell type that mimics Eigen::Matrix<Real, 4, 1> just
// enough for CellTraits to pick it up via the SFINAE on Zero() +
// SizeAtCompileTime, with operator[] for component access and arithmetic.
struct FakeVec4 {
  std::array<Real, 4> a{};
  static constexpr int SizeAtCompileTime = 4;
  static FakeVec4 Zero() { return FakeVec4{}; }
  Real& operator[](int k) { return a[k]; }
  Real operator[](int k) const { return a[k]; }
  FakeVec4 operator*(Real s) const {
    return FakeVec4{{a[0] * s, a[1] * s, a[2] * s, a[3] * s}};
  }
  FakeVec4 operator/(Real s) const { return *this * (Real{1} / s); }
  friend FakeVec4 operator*(Real s, const FakeVec4& v) { return v * s; }
};

TEST_CASE("FluxRegister1D<Real> : store coarse + accumulate fine", "[flux][1d][scalar]") {
  FluxRegister1D<Real> reg;
  reg.reset();
  reg.store_coarse_flux_left(Real{1.5});
  reg.store_coarse_flux_right(Real{-0.5});
  reg.add_fine_flux_left(Real{0.25}, Real{0.1});      // contributes 0.025
  reg.add_fine_flux_left(Real{0.25}, Real{0.1});      // contributes 0.025
  reg.add_fine_flux_right(Real{1.0}, Real{0.2});      // contributes 0.2

  REQUIRE_THAT(reg.coarse_left(),  WithinAbs(1.5, 1e-15));
  REQUIRE_THAT(reg.coarse_right(), WithinAbs(-0.5, 1e-15));
  REQUIRE_THAT(reg.fine_left_avg(Real{0.2}),  WithinAbs(0.25, 1e-15));
  REQUIRE_THAT(reg.fine_right_avg(Real{0.2}), WithinAbs(1.0, 1e-15));
}

TEST_CASE("FluxRegister1D<FakeVec4> : componentwise Kahan add", "[flux][1d][vec]") {
  FluxRegister1D<FakeVec4> reg;
  reg.reset();
  const FakeVec4 F1 = FakeVec4{{1.0, 2.0, 3.0, 4.0}};
  reg.add_fine_flux_left(F1, Real{0.5});      // contributes (0.5, 1, 1.5, 2)
  reg.add_fine_flux_left(F1, Real{0.5});      // contributes (0.5, 1, 1.5, 2)
  const auto avg = reg.fine_left_avg(Real{1.0});
  REQUIRE_THAT(avg[0], WithinAbs(1.0, 1e-15));
  REQUIRE_THAT(avg[1], WithinAbs(2.0, 1e-15));
  REQUIRE_THAT(avg[2], WithinAbs(3.0, 1e-15));
  REQUIRE_THAT(avg[3], WithinAbs(4.0, 1e-15));
}

TEST_CASE("FluxRegister2D<Real> : per-face slots + Kahan accumulate",
          "[flux][2d][scalar]") {
  const pde_core::Index ni = 4, nj = 6;
  FluxRegister2D<Real> reg(ni, nj);
  for (pde_core::Index k = 0; k < nj; ++k) {
    reg.set_F_left_coarse(k, Real{0.1} * static_cast<Real>(k));
  }
  // Add 4 fine contributions per j to F_left, sum should average correctly.
  for (pde_core::Index k = 0; k < nj; ++k) {
    for (int i = 0; i < 4; ++i) reg.add_F_left_fine(k, Real{1.0});
  }
  REQUIRE_THAT(reg.F_left_coarse(2), WithinAbs(0.2, 1e-15));
  // 4 contributions, divide by r^2 = 4 -> avg = 1.0
  REQUIRE_THAT(reg.F_left_fine_avg(0, Real{1} / Real{4}), WithinAbs(1.0, 1e-15));
}

TEST_CASE("FluxRegister2D<FakeVec4> : weighted accumulate (SSPRK3 weights)",
          "[flux][2d][vec][weighted]") {
  const pde_core::Index ni = 2, nj = 2;
  FluxRegister2D<FakeVec4> reg(ni, nj);
  const FakeVec4 F = FakeVec4{{1.0, 1.0, 1.0, 1.0}};
  const Real w1 = 1.0 / 6.0, w2 = 1.0 / 6.0, w3 = 4.0 / 6.0;
  reg.add_F_left_fine_w(0, F, w1);
  reg.add_F_left_fine_w(0, F, w2);
  reg.add_F_left_fine_w(0, F, w3);
  // Sum of weighted contributions = (1/6 + 1/6 + 4/6) * (1,1,1,1) = (1,1,1,1)
  // No division : we read the raw sum via fine_avg with r2_inv = 1.
  const auto v = reg.F_left_fine_avg(0, Real{1});
  REQUIRE_THAT(v[0], WithinAbs(1.0, 1e-15));
  REQUIRE_THAT(v[3], WithinAbs(1.0, 1e-15));
}

TEST_CASE("FluxRegister2D<Real> : per-thread merge equals direct accumulation",
          "[flux][2d][threads]") {
  const pde_core::Index ni = 3, nj = 5;
  FluxRegister2D<Real> reg_direct(ni, nj);
  FluxRegister2D<Real> reg_thread(ni, nj);
  reg_thread.reserve_threads(4);

  // 100 contributions per slot, split round-robin across 4 simulated threads.
  for (int it = 0; it < 100; ++it) {
    const Real val = Real{1.0} + Real{0.001} * static_cast<Real>(it);
    for (pde_core::Index k = 0; k < nj; ++k) {
      reg_direct.add_F_left_fine(k, val);
      reg_thread.add_F_left_fine_thread(k, val, it % 4);
    }
  }
  reg_thread.merge_thread_accumulators();
  for (pde_core::Index k = 0; k < nj; ++k) {
    REQUIRE_THAT(reg_direct.F_left_fine_avg(k, Real{1}),
                 WithinAbs(reg_thread.F_left_fine_avg(k, Real{1}), 1e-13));
  }
}
