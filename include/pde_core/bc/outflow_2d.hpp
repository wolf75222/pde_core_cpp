#pragma once

#include <pde_core/mesh/field2d.hpp>

#include <cstddef>

namespace pde_core {

// Zero-gradient extrapolation on all four sides.
struct OutflowBC2D {
  template <class Cell>
  void apply(Field2D<Cell>& U) const noexcept {
    const auto g  = static_cast<std::ptrdiff_t>(U.ghost());
    const auto nx = static_cast<std::ptrdiff_t>(U.nx());
    const auto ny = static_cast<std::ptrdiff_t>(U.ny());
    for (std::ptrdiff_t j = 0; j < ny; ++j) {
      for (std::ptrdiff_t k = 1; k <= g; ++k) {
        U(-k, j)         = U(0, j);
        U(nx - 1 + k, j) = U(nx - 1, j);
      }
    }
    for (std::ptrdiff_t i = -g; i < nx + g; ++i) {
      for (std::ptrdiff_t k = 1; k <= g; ++k) {
        U(i, -k)         = U(i, 0);
        U(i, ny - 1 + k) = U(i, ny - 1);
      }
    }
  }
};

}  // namespace pde_core
