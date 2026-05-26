#pragma once

#include <pde_core/core/types.hpp>

namespace pde_core {

// Two-dimensional uniform grid on [x_min, x_max] x [y_min, y_max] partitioned
// into nx * ny cells. Cell-centred coordinates as in Domain1D.
struct Domain2D {
  Real x_min;
  Real x_max;
  Real y_min;
  Real y_max;
  Index nx;
  Index ny;

  constexpr Real length_x() const noexcept { return x_max - x_min; }
  constexpr Real length_y() const noexcept { return y_max - y_min; }
  constexpr Real dx() const noexcept { return length_x() / static_cast<Real>(nx); }
  constexpr Real dy() const noexcept { return length_y() / static_cast<Real>(ny); }
  constexpr Real x_cell(Index i) const noexcept {
    return x_min + (static_cast<Real>(i) + Real{0.5}) * dx();
  }
  constexpr Real y_cell(Index j) const noexcept {
    return y_min + (static_cast<Real>(j) + Real{0.5}) * dy();
  }
};

}  // namespace pde_core
