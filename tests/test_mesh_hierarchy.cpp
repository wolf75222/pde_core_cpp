// MeshHierarchy1D and MeshHierarchy2D : single-patch, multi-level, multi-patch.

#include <catch2/catch_test_macros.hpp>

#include <pde_core/amr/mesh_hierarchy1d.hpp>
#include <pde_core/amr/mesh_hierarchy2d.hpp>

using pde_core::MeshHierarchy1D;
using pde_core::MeshHierarchy2D;
using pde_core::Real;

TEST_CASE("MeshHierarchy1D : coarse-only then add fine", "[hierarchy][1d]") {
  MeshHierarchy1D<Real> H(0.0, 1.0, 16, /*ghost=*/2);
  REQUIRE(H.has_fine() == false);
  REQUIRE(H.coarse().nx() == 16);
  H.refine_region(4, 12);
  REQUIRE(H.has_fine() == true);
  REQUIRE(H.fine().nx() == 16);             // r * (12 - 4) = 16
  REQUIRE(H.fine().level() == 1);
  REQUIRE(H.fine_lo_coarse() == 4);
  REQUIRE(H.fine_hi_coarse() == 12);
  H.clear_refinement();
  REQUIRE(H.has_fine() == false);
}

TEST_CASE("MeshHierarchy2D : legacy two-level path", "[hierarchy][2d][legacy]") {
  MeshHierarchy2D<Real> H(0.0, 1.0, 0.0, 1.0, 16, 16, 2);
  REQUIRE(H.has_fine() == false);
  REQUIRE(H.coarse().nx() == 16);
  REQUIRE(H.coarse().ny() == 16);
  H.refine_region(4, 12, 6, 10);
  REQUIRE(H.has_fine() == true);
  REQUIRE(H.fine().nx() == 16);
  REQUIRE(H.fine().ny() == 8);
  REQUIRE(H.fine_lo_i_coarse() == 4);
  REQUIRE(H.fine_hi_i_coarse() == 12);
  REQUIRE(H.fine_lo_j_coarse() == 6);
  REQUIRE(H.fine_hi_j_coarse() == 10);
}

TEST_CASE("MeshHierarchy2D : multi-level single-patch", "[hierarchy][2d][multilevel]") {
  MeshHierarchy2D<Real> H(0.0, 1.0, 0.0, 1.0, 16, 16, 2);
  H.refine_at_level(1, 4, 12, 4, 12);
  H.refine_at_level(2, 2, 6, 2, 6);
  REQUIRE(H.finest_level() == 2);
  REQUIRE(H.is_refined(1));
  REQUIRE(H.is_refined(2));
  REQUIRE(H.n_patches(1) == 1);
  REQUIRE(H.n_patches(2) == 1);
  REQUIRE(H.block(1).level() == 1);
  REQUIRE(H.block(2).level() == 2);
  // Block at level 2 has dimensions r * (6 - 2) = 8 cells per side.
  REQUIRE(H.block(2).nx() == 8);
  REQUIRE(H.block(2).ny() == 8);
}

TEST_CASE("MeshHierarchy2D : multi-patch at level 1", "[hierarchy][2d][multipatch]") {
  MeshHierarchy2D<Real> H(0.0, 1.0, 0.0, 1.0, 16, 16, 2);
  using Cov = MeshHierarchy2D<Real>::Coverage;
  H.set_patches_at_level(1, {Cov{2, 6, 2, 6}, Cov{10, 14, 10, 14}});
  REQUIRE(H.n_patches(1) == 2);
  REQUIRE(H.patch(1, 0).nx() == 8);   // r * (6 - 2)
  REQUIRE(H.patch(1, 1).nx() == 8);
  REQUIRE(H.patch_coverage(1, 0).i_lo == 2);
  REQUIRE(H.patch_coverage(1, 1).i_lo == 10);
  // add_patch_at_level appends without dropping existing patches at L
  H.add_patch_at_level(1, Cov{2, 6, 10, 14});
  REQUIRE(H.n_patches(1) == 3);
}

TEST_CASE("MeshHierarchy2D : set_patches_at_level wipes deeper levels",
          "[hierarchy][2d][cleanup]") {
  MeshHierarchy2D<Real> H(0.0, 1.0, 0.0, 1.0, 16, 16, 2);
  H.refine_at_level(1, 0, 16, 0, 16);
  H.refine_at_level(2, 0, 8, 0, 8);
  REQUIRE(H.n_patches(2) == 1);
  using Cov = MeshHierarchy2D<Real>::Coverage;
  // Reset level 1 : level 2 should be wiped (orphan child)
  H.set_patches_at_level(1, {Cov{4, 12, 4, 12}});
  REQUIRE(H.n_patches(1) == 1);
  REQUIRE(H.n_patches(2) == 0);
}
