// Berger-Rigoutsos clustering : verify on a few hand-built mark grids.

#include <catch2/catch_test_macros.hpp>

#include <pde_core/amr/clustering2d.hpp>

#include <vector>

using pde_core::Box2D;
using pde_core::ClusteringParams2D;
using pde_core::cluster_berger_rigoutsos;

TEST_CASE("Empty mark grid -> no boxes", "[clustering][empty]") {
  std::vector<bool> marks(8 * 8, false);
  auto boxes = cluster_berger_rigoutsos(marks, 8, 8);
  REQUIRE(boxes.empty());
}

TEST_CASE("Single block of tags -> one box exactly covering it",
          "[clustering][single]") {
  const std::ptrdiff_t nx = 16, ny = 16;
  std::vector<bool> marks(static_cast<std::size_t>(nx * ny), false);
  for (std::ptrdiff_t j = 4; j < 10; ++j)
    for (std::ptrdiff_t i = 5; i < 12; ++i)
      marks[static_cast<std::size_t>(j * nx + i)] = true;
  auto boxes = cluster_berger_rigoutsos(marks, nx, ny);
  REQUIRE(boxes.size() == 1);
  const Box2D& b = boxes.front();
  REQUIRE(b.i_lo == 5);
  REQUIRE(b.i_hi == 12);
  REQUIRE(b.j_lo == 4);
  REQUIRE(b.j_hi == 10);
}

TEST_CASE("Two disjoint blocks -> split on the hole",
          "[clustering][split]") {
  const std::ptrdiff_t nx = 24, ny = 8;
  std::vector<bool> marks(static_cast<std::size_t>(nx * ny), false);
  for (std::ptrdiff_t j = 2; j < 6; ++j) {
    for (std::ptrdiff_t i = 2; i < 6;  ++i) marks[static_cast<std::size_t>(j * nx + i)] = true;
    for (std::ptrdiff_t i = 16; i < 20; ++i) marks[static_cast<std::size_t>(j * nx + i)] = true;
  }
  ClusteringParams2D p;
  p.efficiency_threshold = 0.7;
  p.min_box_size = 2;
  auto boxes = cluster_berger_rigoutsos(marks, nx, ny, p);
  REQUIRE(boxes.size() == 2);
  // Boxes cover the two clusters separately.
  std::ptrdiff_t total_area = 0;
  for (const auto& b : boxes) {
    total_area += (b.i_hi - b.i_lo) * (b.j_hi - b.j_lo);
  }
  REQUIRE(total_area == 32);  // 2 * 4 * 4
}
