// ghost_fill2d : two-level multi-patch ghost fill via parent injection
// and sibling override.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <pde_core/amr/ghost_fill2d.hpp>
#include <pde_core/amr/mesh_hierarchy2d.hpp>
#include <pde_core/mesh/field2d.hpp>

using Catch::Matchers::WithinAbs;
using pde_core::Field2D;
using pde_core::MeshHierarchy2D;
using pde_core::Real;
using pde_core::fill_patch_ghosts_multipatch;

TEST_CASE("ghost_fill : single patch L=1 fills ghosts from coarse",
          "[ghost_fill][1patch]") {
  MeshHierarchy2D<Real> mesh(0.0, 1.0, 0.0, 1.0, 16, 16);
  // Fill the coarse field with a linear ramp U(i,j) = i (incl. ghosts).
  for (std::ptrdiff_t j = -2; j < 18; ++j)
    for (std::ptrdiff_t i = -2; i < 18; ++i)
      mesh.coarse().field()(i, j) = static_cast<Real>(i);

  mesh.refine_region(4, 12, 4, 12);  // single patch at L=1
  Field2D<Real> parent_n  = mesh.coarse().field();
  Field2D<Real> parent_np1 = mesh.coarse().field();  // same time level

  fill_patch_ghosts_multipatch(mesh, /*L=*/1, /*patch=*/0,
                                parent_n, parent_np1, /*alpha=*/0.0);
  // Patch at L=1 has box i_lo = 8 (= 2 * 4). Fine cell (0, 0) -> coarse (4, 4).
  // Its left ghost at i_local = -1 should map to coarse cell (3, 4) = 3.
  const auto& fine = mesh.patch(1, 0).field();
  REQUIRE_THAT(fine(-1, 0), WithinAbs(Real{3}, 1e-12));
  REQUIRE_THAT(fine(-2, 0), WithinAbs(Real{3}, 1e-12));
  // Right ghost should map to coarse(12, 4) = 12.
  const auto nxf = static_cast<std::ptrdiff_t>(fine.nx());
  REQUIRE_THAT(fine(nxf,     0), WithinAbs(Real{12}, 1e-12));
  REQUIRE_THAT(fine(nxf + 1, 0), WithinAbs(Real{12}, 1e-12));
}

TEST_CASE("ghost_fill : sibling override prefers same-level neighbour",
          "[ghost_fill][sibling]") {
  MeshHierarchy2D<Real> mesh(0.0, 1.0, 0.0, 1.0, 16, 16);
  // Coarse uniform = 0.
  for (std::ptrdiff_t j = -2; j < 18; ++j)
    for (std::ptrdiff_t i = -2; i < 18; ++i)
      mesh.coarse().field()(i, j) = Real{0.0};

  // Two adjacent patches at L=1 : i = [4, 8) and i = [8, 12), both j = [4, 8)
  std::vector<MeshHierarchy2D<Real>::Coverage> covs;
  covs.push_back({4, 8, 4, 8});
  covs.push_back({8, 12, 4, 8});
  mesh.set_patches_at_level(1, covs);

  // Mark the second patch's interior with distinctive values.
  auto& sib = mesh.patch(1, 1).field();
  const auto nxs = static_cast<std::ptrdiff_t>(sib.nx());
  const auto nys = static_cast<std::ptrdiff_t>(sib.ny());
  for (std::ptrdiff_t j = 0; j < nys; ++j)
    for (std::ptrdiff_t i = 0; i < nxs; ++i)
      sib(i, j) = Real{99.0};

  Field2D<Real> parent_n  = mesh.coarse().field();
  Field2D<Real> parent_np1 = mesh.coarse().field();
  fill_patch_ghosts_multipatch(mesh, 1, 0, parent_n, parent_np1, 0.0);

  // Patch 0's right ghost (at i_local = nxp0) sits at L-index 8 = sibling's
  // i_lo. The fill should grab the sibling's first interior cell = 99.
  const auto& p0 = mesh.patch(1, 0).field();
  const auto nxp0 = static_cast<std::ptrdiff_t>(p0.nx());
  REQUIRE_THAT(p0(nxp0, 0), WithinAbs(Real{99}, 1e-12));
}
