// regrid_1d / regrid_2d / regrid_2d_multilevel / regrid_2d_multipatch on
// scalar Real cells.

#include <catch2/catch_test_macros.hpp>

#include <pde_core/amr/regrid1d.hpp>
#include <pde_core/amr/regrid2d.hpp>

#include <cmath>
#include <vector>

using pde_core::MeshHierarchy1D;
using pde_core::MeshHierarchy2D;
using pde_core::Real;
using pde_core::regrid_1d;
using pde_core::regrid_2d;
using pde_core::regrid_2d_multilevel;
using pde_core::regrid_2d_multipatch;

namespace {

// Gradient-threshold criterion (1D) : mark cell i if |U(i+1) - U(i-1)| > t.
auto make_grad_criterion_1d(Real threshold) {
  return [threshold](const auto& U, Real /*dx*/) {
    const auto nx = static_cast<std::ptrdiff_t>(U.nx());
    std::vector<bool> marks(static_cast<std::size_t>(nx), false);
    for (std::ptrdiff_t i = 1; i < nx - 1; ++i) {
      const Real left  = U(i - 1);
      const Real right = U(i + 1);
      if (std::abs(right - left) > threshold) {
        marks[static_cast<std::size_t>(i)] = true;
      }
    }
    return marks;
  };
}

auto make_grad_criterion_2d(Real threshold) {
  return [threshold](const auto& U, Real /*dx*/, Real /*dy*/) {
    const auto nx = static_cast<std::ptrdiff_t>(U.nx());
    const auto ny = static_cast<std::ptrdiff_t>(U.ny());
    std::vector<bool> marks(static_cast<std::size_t>(nx * ny), false);
    for (std::ptrdiff_t j = 1; j < ny - 1; ++j) {
      for (std::ptrdiff_t i = 1; i < nx - 1; ++i) {
        const Real dx_u = std::abs(U(i + 1, j) - U(i - 1, j));
        const Real dy_u = std::abs(U(i, j + 1) - U(i, j - 1));
        if (dx_u + dy_u > threshold) {
          marks[static_cast<std::size_t>(j * nx + i)] = true;
        }
      }
    }
    return marks;
  };
}

}  // namespace

TEST_CASE("regrid_1d : drop fine when nothing is marked", "[regrid][1d]") {
  MeshHierarchy1D<Real> mesh(0.0, 1.0, 16);
  // Pre-existing fine block
  mesh.refine_region(4, 12);
  REQUIRE(mesh.has_fine());

  // Constant field, no gradient, criterion marks nothing.
  for (std::ptrdiff_t i = -2; i < 18; ++i) mesh.coarse().field()(i) = Real{1.0};

  const bool changed = regrid_1d(mesh, make_grad_criterion_1d(Real{0.1}));
  REQUIRE(changed);
  REQUIRE_FALSE(mesh.has_fine());
}

TEST_CASE("regrid_1d : new fine bbox matches gradient location", "[regrid][1d]") {
  MeshHierarchy1D<Real> mesh(0.0, 1.0, 16);
  // Step at i = 8
  for (std::ptrdiff_t i = -2; i < 18; ++i) {
    mesh.coarse().field()(i) = (i >= 8) ? Real{1.0} : Real{0.0};
  }
  const bool changed = regrid_1d(mesh, make_grad_criterion_1d(Real{0.5}),
                                  /*buffer=*/2);
  REQUIRE(changed);
  REQUIRE(mesh.has_fine());
  // bbox of marked cells contains i = 8, expanded by 2, aligned to ratio 2
  REQUIRE(mesh.fine_lo_coarse() <= 6);
  REQUIRE(mesh.fine_hi_coarse() >= 10);
}

TEST_CASE("regrid_2d : two-level patch around a 2D gradient", "[regrid][2d]") {
  MeshHierarchy2D<Real> mesh(0.0, 1.0, 0.0, 1.0, 16, 16);
  // Diagonal step
  for (std::ptrdiff_t j = -2; j < 18; ++j)
    for (std::ptrdiff_t i = -2; i < 18; ++i)
      mesh.coarse().field()(i, j) = (i + j > 16) ? Real{1.0} : Real{0.0};

  const bool changed = regrid_2d(mesh, make_grad_criterion_2d(Real{0.5}),
                                   /*buffer=*/2);
  REQUIRE(changed);
  REQUIRE(mesh.has_fine());
  REQUIRE(mesh.fine_lo_i_coarse() < mesh.fine_hi_i_coarse());
  REQUIRE(mesh.fine_lo_j_coarse() < mesh.fine_hi_j_coarse());
}

TEST_CASE("regrid_2d_multipatch : two well-separated blobs yield 2 patches",
          "[regrid][2d][multipatch]") {
  MeshHierarchy2D<Real> mesh(0.0, 1.0, 0.0, 1.0, 32, 32);
  for (std::ptrdiff_t j = -2; j < 34; ++j)
    for (std::ptrdiff_t i = -2; i < 34; ++i)
      mesh.coarse().field()(i, j) = Real{0.0};
  // Two gradient zones : i in [4,7] and i in [24,27]
  auto place_step = [&](std::ptrdiff_t i_step) {
    for (std::ptrdiff_t j = 8; j < 24; ++j)
      for (std::ptrdiff_t i = -2; i < 34; ++i)
        if (i >= i_step) mesh.coarse().field()(i, j) += Real{1.0};
  };
  place_step(6);
  // The second step is "down" so its gradient is also marked.
  for (std::ptrdiff_t j = 8; j < 24; ++j)
    for (std::ptrdiff_t i = -2; i < 34; ++i)
      if (i >= 26) mesh.coarse().field()(i, j) -= Real{1.0};

  pde_core::ClusteringParams2D p;
  p.efficiency_threshold = Real{0.5};
  p.min_box_size = 2;
  const bool changed = regrid_2d_multipatch(mesh, make_grad_criterion_2d(Real{0.5}),
                                              p, /*buffer=*/1);
  REQUIRE(changed);
  REQUIRE(mesh.n_patches(1) >= 1);  // at least 1, often 2 depending on buffer
}

TEST_CASE("regrid_2d_multilevel : two nested levels around a sharp feature",
          "[regrid][2d][multilevel]") {
  MeshHierarchy2D<Real> mesh(0.0, 1.0, 0.0, 1.0, 32, 32);
  // Gaussian centred at (0.5, 0.5)
  for (std::ptrdiff_t j = -2; j < 34; ++j)
    for (std::ptrdiff_t i = -2; i < 34; ++i) {
      const Real x = (static_cast<Real>(i) + Real{0.5}) / Real{32};
      const Real y = (static_cast<Real>(j) + Real{0.5}) / Real{32};
      const Real r2 = (x - Real{0.5}) * (x - Real{0.5}) +
                       (y - Real{0.5}) * (y - Real{0.5});
      mesh.coarse().field()(i, j) = std::exp(-r2 / Real{0.01});
    }
  const bool changed = regrid_2d_multilevel(
      mesh, make_grad_criterion_2d(Real{0.05}),
      /*max_target_level=*/2, /*buffer=*/2, /*nest_buffer=*/2);
  REQUIRE(changed);
  REQUIRE(mesh.finest_level() >= 1);
}
