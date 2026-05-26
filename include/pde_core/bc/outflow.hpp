#pragma once

#include <pde_core/mesh/field1d.hpp>

#include <cstddef>

namespace pde_core {

// Transmissive (zero-gradient) outflow BC. U_ghost = U_interior_boundary.
// Lets waves leave the domain without reflection in the limit where the
// boundary stays far from the active region for the duration of the run.
// Fine for Riemann problem tests, NOT a true non-reflecting BC (for that
// see characteristic-based Sommerfeld variants in the consumer solvers).
struct OutflowBC1D {
  template <class Cell>
  void apply(Field1D<Cell>& U) const noexcept {
    const auto nx = static_cast<std::ptrdiff_t>(U.nx());
    const auto g  = static_cast<std::ptrdiff_t>(U.ghost());
    for (std::ptrdiff_t k = 1; k <= g; ++k) {
      U(-k)         = U(0);
      U(nx - 1 + k) = U(nx - 1);
    }
  }
};

}  // namespace pde_core
