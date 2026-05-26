#pragma once

#include <pde_core/core/types.hpp>

#include <type_traits>

namespace pde_core {

// Trait describing how to interact with a cell payload type. The default
// fallback assumes Cell is a scalar (Real, float, double, int) and treats
// it as a 1-component value. The specialisation below detects Eigen-style
// types (those exposing a static Zero() and a compile-time SizeAtCompileTime)
// and routes through their component accessors instead.
//
// Two facets :
//   - zero()       : value used to initialise a fresh Field<Cell> buffer.
//   - size()       : number of scalar components in Cell.
//   - component(c, k) / component(const c, k) : access the k-th scalar
//     component as a Real reference. For scalar Cell, k is ignored.
//
// Why we need the component accessors : Kahan-compensated summation in the
// flux registers needs to manipulate the lost low-order bits one scalar at
// a time, with a `volatile` cast to defeat the compiler's
// reassociation optimisations. Eigen expression templates would otherwise
// fuse the operations and skip the compensation step. By going through
// CellTraits<Cell>::component(c, k), the same Kahan add works for both
// `Real` (1 component) and `Eigen::Matrix<Real, N, 1>` (N components),
// with the loop unrolled at compile time when SizeAtCompileTime is known.

template <class Cell, class = void>
struct CellTraits {
  static constexpr Cell zero() noexcept(noexcept(Cell{})) { return Cell{}; }
  static constexpr int size() noexcept { return 1; }
  static constexpr Real& component(Cell& c, int /*k*/) noexcept { return c; }
  static constexpr Real component(const Cell& c, int /*k*/) noexcept { return c; }
};

// Specialisation for Eigen-style types : detected by the presence of a
// static Zero() method. The trait then routes through Eigen's compile-time
// size and bracket-access operator.
template <class Cell>
struct CellTraits<Cell,
                   std::void_t<decltype(Cell::Zero()),
                                decltype(Cell::SizeAtCompileTime)>> {
  static auto zero() noexcept(noexcept(Cell::Zero())) {
    return Cell::Zero();
  }
  static constexpr int size() noexcept {
    return static_cast<int>(Cell::SizeAtCompileTime);
  }
  static Real& component(Cell& c, int k) noexcept {
    return c[k];
  }
  static Real component(const Cell& c, int k) noexcept {
    return c[k];
  }
};

// Convenience helpers. Prefer these at call sites :
//   cell_zero<Cell>()        instead of CellTraits<Cell>::zero()
//   cell_size<Cell>()        instead of CellTraits<Cell>::size()
//   cell_at(cell, k)         instead of CellTraits<Cell>::component(cell, k)
template <class Cell>
inline auto cell_zero() noexcept(noexcept(CellTraits<Cell>::zero())) {
  return CellTraits<Cell>::zero();
}

template <class Cell>
inline constexpr int cell_size() noexcept {
  return CellTraits<Cell>::size();
}

template <class Cell>
inline Real& cell_at(Cell& c, int k) noexcept {
  return CellTraits<Cell>::component(c, k);
}

template <class Cell>
inline Real cell_at(const Cell& c, int k) noexcept {
  return CellTraits<Cell>::component(c, k);
}

// Kahan-compensated add `sum += x`, with `cmp` carrying the running
// compensation term. Works component-wise so the same routine handles
// scalar and Eigen-vector cells. The `volatile` cast on the temporary
// blocks reassociation optimisations that would otherwise defeat the
// compensation (Kahan 1965, "Pracniques: further remarks on reducing
// truncation errors", Communications of the ACM 8(1):40).
template <class Cell>
inline void kahan_add(Cell& sum, Cell& cmp, const Cell& x) noexcept {
  const int n = cell_size<Cell>();
  for (int k = 0; k < n; ++k) {
    const Real y = cell_at(x, k) - cell_at(cmp, k);
    volatile Real t = cell_at(sum, k) + y;
    cell_at(cmp, k) = (static_cast<Real>(t) - cell_at(sum, k)) - y;
    cell_at(sum, k) = static_cast<Real>(t);
  }
}

}  // namespace pde_core
