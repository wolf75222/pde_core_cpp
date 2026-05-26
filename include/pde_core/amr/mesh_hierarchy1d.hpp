#pragma once

#include <pde_core/amr/box1d.hpp>
#include <pde_core/amr/mesh_block1d.hpp>
#include <pde_core/core/types.hpp>

#include <cmath>
#include <cstddef>
#include <optional>

namespace pde_core {

// 1D mesh hierarchy with refinement ratio 2, supporting two levels (0 and 1)
// and a single block per level. Static refinement : the fine patch is set
// once at construction or via refine_region. Templated on the per-cell
// payload Cell, so that the hierarchy works equally for scalar problems
// (Cell = Real) and for systems (Cell = Eigen::Matrix<Real, N, 1>).
template <class Cell>
class MeshHierarchy1D {
 public:
  static constexpr int max_level = 1;
  static constexpr Index refinement_ratio = 2;

  // Construct with a single coarse block covering [x_min, x_max] on level 0.
  MeshHierarchy1D(Real x_min, Real x_max, Index nx_base, Index ghost = 2)
      : x_min_(x_min),
        x_max_(x_max),
        nx_base_(nx_base),
        ghost_(ghost),
        coarse_(Box1D{0, static_cast<std::ptrdiff_t>(nx_base), 0}, ghost) {}

  // Add a single fine block covering coarse cells [i_lo, i_hi) on level 0.
  // The resulting level-1 block has index range [r * i_lo, r * i_hi).
  // Calling refine_region again replaces the existing fine block.
  void refine_region(std::ptrdiff_t i_lo_coarse, std::ptrdiff_t i_hi_coarse) {
    const auto r = static_cast<std::ptrdiff_t>(refinement_ratio);
    fine_present_ = true;
    fine_lo_coarse_ = i_lo_coarse;
    fine_hi_coarse_ = i_hi_coarse;
    fine_block_.emplace(Box1D{r * i_lo_coarse, r * i_hi_coarse, 1}, ghost_);
  }

  // Drop the fine block. Used by dynamic regridding when no cells need
  // refinement.
  void clear_refinement() noexcept {
    fine_present_ = false;
    fine_block_.reset();
  }

  // Geometry
  Real x_min() const noexcept { return x_min_; }
  Real x_max() const noexcept { return x_max_; }
  Index nx_base() const noexcept { return nx_base_; }
  Real dx(int level) const noexcept {
    const Real base = (x_max_ - x_min_) / static_cast<Real>(nx_base_);
    return base / std::pow(static_cast<Real>(refinement_ratio), level);
  }
  Real x_cell(int level, std::ptrdiff_t i) const noexcept {
    return x_min_ + (static_cast<Real>(i) + Real{0.5}) * dx(level);
  }

  // Blocks
  MeshBlock1D<Cell>& coarse() noexcept { return coarse_; }
  const MeshBlock1D<Cell>& coarse() const noexcept { return coarse_; }
  bool has_fine() const noexcept { return fine_present_; }
  MeshBlock1D<Cell>& fine() noexcept { return *fine_block_; }
  const MeshBlock1D<Cell>& fine() const noexcept { return *fine_block_; }

  // Coarse-index bounds of the fine patch (inclusive lower, exclusive upper).
  std::ptrdiff_t fine_lo_coarse() const noexcept { return fine_lo_coarse_; }
  std::ptrdiff_t fine_hi_coarse() const noexcept { return fine_hi_coarse_; }

 private:
  Real x_min_, x_max_;
  Index nx_base_;
  Index ghost_;
  MeshBlock1D<Cell> coarse_;
  bool fine_present_ = false;
  std::ptrdiff_t fine_lo_coarse_ = 0;
  std::ptrdiff_t fine_hi_coarse_ = 0;
  std::optional<MeshBlock1D<Cell>> fine_block_;
};

}  // namespace pde_core
