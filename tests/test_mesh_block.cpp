// MeshBlock1D / MeshBlock2D wrap a Box + a Field.

#include <catch2/catch_test_macros.hpp>

#include <pde_core/amr/box1d.hpp>
#include <pde_core/amr/box2d.hpp>
#include <pde_core/amr/mesh_block1d.hpp>
#include <pde_core/amr/mesh_block2d.hpp>

using pde_core::Box1D;
using pde_core::Box2D;
using pde_core::MeshBlock1D;
using pde_core::MeshBlock2D;
using pde_core::Real;

TEST_CASE("MeshBlock1D<Real> : box + field", "[meshblock][1d]") {
  Box1D box{2, 14, /*level=*/1};
  MeshBlock1D<Real> blk(box, /*ghost=*/2);
  REQUIRE(blk.nx() == 12);
  REQUIRE(blk.level() == 1);
  REQUIRE(blk.field().nx() == 12);
  REQUIRE(blk.field().ghost() == 2);
  blk.field()(0) = Real{1.5};
  blk.field()(11) = Real{2.5};
  REQUIRE(blk.field()(0) == Real{1.5});
  REQUIRE(blk.field()(11) == Real{2.5});
}

TEST_CASE("MeshBlock2D<Real> : box + field", "[meshblock][2d]") {
  Box2D box{0, 8, 0, 6, /*level=*/0};
  MeshBlock2D<Real> blk(box, /*ghost=*/1);
  REQUIRE(blk.nx() == 8);
  REQUIRE(blk.ny() == 6);
  REQUIRE(blk.level() == 0);
  REQUIRE(blk.field().nx() == 8);
  REQUIRE(blk.field().ny() == 6);
  REQUIRE(blk.field().ghost() == 1);
  blk.field()(3, 4) = Real{0.75};
  REQUIRE(blk.field()(3, 4) == Real{0.75});
}
