#pragma once

#include <pde_core/core/types.hpp>

#include <cstddef>

namespace pde_core {

// Index-space interval [i_lo, i_hi) at a refinement level. Pure geometry,
// no data. Convention : a box at level l uses indices on the level-l grid,
// whose physical extent is i * dx_l with dx_l = dx_0 / r^l (r = refinement
// ratio, fixed at 2 in this build).
struct Box1D {
  std::ptrdiff_t i_lo;
  std::ptrdiff_t i_hi;
  int level;

  constexpr Index nx() const noexcept {
    return static_cast<Index>(i_hi - i_lo);
  }
};

}  // namespace pde_core
