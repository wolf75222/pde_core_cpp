#pragma once

#include <pde_core/mesh/field2d.hpp>

#include <cstddef>

namespace pde_core {

// Periodic in both x and y. East-West strips are filled first so the
// subsequent North-South pass fills the four corners correctly.
struct PeriodicBC2D {
  template <class Cell>
  void apply(Field2D<Cell>& U) const noexcept {
    const auto g  = static_cast<std::ptrdiff_t>(U.ghost());
    const auto nx = static_cast<std::ptrdiff_t>(U.nx());
    const auto ny = static_cast<std::ptrdiff_t>(U.ny());
    // East-West first.
    for (std::ptrdiff_t j = 0; j < ny; ++j) {
      for (std::ptrdiff_t k = 1; k <= g; ++k) {
        U(-k, j)         = U(nx - k, j);
        U(nx - 1 + k, j) = U(k - 1, j);
      }
    }
    // Then North-South over the full extended span so corners get filled.
    for (std::ptrdiff_t i = -g; i < nx + g; ++i) {
      for (std::ptrdiff_t k = 1; k <= g; ++k) {
        U(i, -k)         = U(i, ny - k);
        U(i, ny - 1 + k) = U(i, k - 1);
      }
    }
  }
};

}  // namespace pde_core
