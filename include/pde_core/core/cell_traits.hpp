#pragma once

#include <pde_core/core/types.hpp>

#include <type_traits>

namespace pde_core {

// Trait for "what is the zero value of this cell type". Used by Field*D to
// initialise data without committing to a fixed cell type.
//
// Default : value-initialisation (yields Real{0}, ConservedState::Zero() does
// not match this fallback, so Eigen matrix types need an explicit
// specialisation below).
//
// To plug a new cell type into pde_core, either ensure its default
// constructor zero-initialises, or specialise this trait in your consumer
// header before instantiating Field*D<Cell>.
template <class Cell, class = void>
struct CellTraits {
  static constexpr Cell zero() noexcept(noexcept(Cell{})) { return Cell{}; }
};

// Specialisation : if Cell has a static `Zero()` method (matches the
// Eigen::Matrix and Eigen::Array families), use it. SFINAE on the
// expression `Cell::Zero()`.
template <class Cell>
struct CellTraits<Cell,
                   std::void_t<decltype(Cell::Zero())>> {
  static auto zero() noexcept(noexcept(Cell::Zero())) {
    return Cell::Zero();
  }
};

// Convenience helper : `cell_zero<Cell>()` is the idiomatic call site.
template <class Cell>
inline auto cell_zero() noexcept(noexcept(CellTraits<Cell>::zero())) {
  return CellTraits<Cell>::zero();
}

}  // namespace pde_core
