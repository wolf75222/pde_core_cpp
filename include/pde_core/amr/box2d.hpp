#pragma once

#include <pde_core/core/types.hpp>

#include <cstddef>

namespace pde_core {

// 2D index-space rectangle [i_lo, i_hi) x [j_lo, j_hi) at a refinement
// level. Tensor-product extension of Box1D.
struct Box2D {
  std::ptrdiff_t i_lo;
  std::ptrdiff_t i_hi;
  std::ptrdiff_t j_lo;
  std::ptrdiff_t j_hi;
  int level;

  constexpr Index nx() const noexcept {
    return static_cast<Index>(i_hi - i_lo);
  }
  constexpr Index ny() const noexcept {
    return static_cast<Index>(j_hi - j_lo);
  }
};

}  // namespace pde_core
