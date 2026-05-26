#pragma once

#include <pde_core/core/types.hpp>

namespace pde_core {

// One-dimensional uniform grid on [x_min, x_max] partitioned into nx cells.
// Cells are indexed 0 .. nx-1 with cell-centred coordinates
// x_cell(i) = x_min + (i + 1/2) dx.
struct Domain1D {
  Real x_min;
  Real x_max;
  Index nx;

  constexpr Real length() const noexcept { return x_max - x_min; }
  constexpr Real dx() const noexcept { return length() / static_cast<Real>(nx); }
  constexpr Real x_cell(Index i) const noexcept {
    return x_min + (static_cast<Real>(i) + Real{0.5}) * dx();
  }
};

}  // namespace pde_core
