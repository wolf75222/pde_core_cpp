#pragma once

#include <pde_core/mesh/field1d.hpp>

#include <cstddef>

namespace pde_core {

// Periodic boundary on [x_min, x_max] : U(-k) = U(nx - k) and
// U(nx - 1 + k) = U(k - 1) for k = 1 .. ghost. Pure index manipulation,
// works for any cell type.
struct PeriodicBC1D {
  template <class Cell>
  void apply(Field1D<Cell>& U) const noexcept {
    const auto nx = static_cast<std::ptrdiff_t>(U.nx());
    const auto g  = static_cast<std::ptrdiff_t>(U.ghost());
    for (std::ptrdiff_t k = 1; k <= g; ++k) {
      U(-k) = U(nx - k);
      U(nx - 1 + k) = U(k - 1);
    }
  }
};

}  // namespace pde_core
