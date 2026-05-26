#pragma once

#include <pde_core/core/cell_traits.hpp>
#include <pde_core/core/types.hpp>

namespace pde_core {

// FluxRegister1D : flux-mismatch bookkeeping for the two coarse-fine
// interfaces bordering a single fine block.
//
// During a coarse step subcycled at the fine level, the coarse cells next
// to the fine patch use a coarse face flux. The fine patch advances over
// r = 2 sub-steps using its own (more accurate) face fluxes. At the end of
// the coarse cycle the reflux correction restores discrete conservation by
// replacing the coarse-side estimate with the time-averaged sum of fine
// fluxes (Berger and Colella 1989).
//
// Templated on the cell payload Cell : Real for scalar conservation laws,
// Eigen::Matrix<Real, N, 1> for systems. The Kahan-compensated add is
// routed through CellTraits<Cell>::component so the same code path works
// for both, with the loop unrolled at compile time.
//
// The fine-side accumulation uses Kahan compensated summation so that the
// reflux correction is bit-reproducible across compilers and architectures
// (otherwise the per-add O(eps) error builds up across multiple AMR steps
// and shows up as a small platform-visible mass drift).
template <class Cell>
class FluxRegister1D {
 public:
  void reset() noexcept {
    F_coarse_left_ = cell_zero<Cell>();
    F_coarse_right_ = cell_zero<Cell>();
    F_fine_left_accum_ = cell_zero<Cell>();
    F_fine_right_accum_ = cell_zero<Cell>();
    F_fine_left_cmp_ = cell_zero<Cell>();
    F_fine_right_cmp_ = cell_zero<Cell>();
  }

  void store_coarse_flux_left(const Cell& F) noexcept {
    F_coarse_left_ = F;
  }
  void store_coarse_flux_right(const Cell& F) noexcept {
    F_coarse_right_ = F;
  }

  void add_fine_flux_left(const Cell& F, Real dt_fine) noexcept {
    const Cell wF = F * dt_fine;
    kahan_add(F_fine_left_accum_, F_fine_left_cmp_, wF);
  }
  void add_fine_flux_right(const Cell& F, Real dt_fine) noexcept {
    const Cell wF = F * dt_fine;
    kahan_add(F_fine_right_accum_, F_fine_right_cmp_, wF);
  }

  const Cell& coarse_left() const noexcept { return F_coarse_left_; }
  const Cell& coarse_right() const noexcept { return F_coarse_right_; }
  Cell fine_left_avg(Real dt_coarse) const noexcept {
    return F_fine_left_accum_ / dt_coarse;
  }
  Cell fine_right_avg(Real dt_coarse) const noexcept {
    return F_fine_right_accum_ / dt_coarse;
  }

 private:
  Cell F_coarse_left_ = cell_zero<Cell>();
  Cell F_coarse_right_ = cell_zero<Cell>();
  Cell F_fine_left_accum_ = cell_zero<Cell>();
  Cell F_fine_right_accum_ = cell_zero<Cell>();
  Cell F_fine_left_cmp_ = cell_zero<Cell>();
  Cell F_fine_right_cmp_ = cell_zero<Cell>();
};

}  // namespace pde_core
