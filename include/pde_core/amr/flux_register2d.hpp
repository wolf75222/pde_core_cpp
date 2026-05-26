#pragma once

#include <pde_core/core/cell_traits.hpp>
#include <pde_core/core/types.hpp>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace pde_core {

// FluxRegister2D : flux-mismatch bookkeeping for the four coarse-fine
// interfaces bordering a single fine block in 2D.
//
// Each coarse face along an interface is matched, at the fine resolution,
// by r = 2 stacked fine faces, accumulated over r time sub-steps, so each
// coarse-face slot receives r * r raw fine flux contributions during one
// coarse step. The integrator divides by r * r at reflux time to get the
// fine-side time-and-space average.
//
// Templated on the cell payload Cell : Real for scalar (advection,
// Burgers), Eigen::Matrix<Real, N, 1> for systems (Euler N = 4).
//
// Fine-side and weighted coarse-side accumulations use Kahan compensated
// summation routed through CellTraits<Cell>::component. The same code
// path covers scalar and Eigen-vector cells, with the inner loop
// unrolled at compile time when Cell has a fixed SizeAtCompileTime.
//
// Per-thread accumulators : the integrators' hot loop wraps perimeter
// flux register writes in PDE_OMP_CRITICAL to avoid races. With many
// threads the critical section becomes a bottleneck. Each thread can
// instead accumulate into its own slice and the canonical accumulators
// are merged after the parallel region. Call reserve_threads(n_threads)
// once before the parallel for, then merge_thread_accumulators() after.
template <class Cell>
class FluxRegister2D {
 public:
  FluxRegister2D(Index ni_coarse, Index nj_coarse)
      : ni_(ni_coarse), nj_(nj_coarse),
        F_left_c_(nj_coarse, cell_zero<Cell>()),
        F_right_c_(nj_coarse, cell_zero<Cell>()),
        F_left_c_cmp_(nj_coarse, cell_zero<Cell>()),
        F_right_c_cmp_(nj_coarse, cell_zero<Cell>()),
        F_left_f_sum_(nj_coarse, cell_zero<Cell>()),
        F_right_f_sum_(nj_coarse, cell_zero<Cell>()),
        F_left_f_cmp_(nj_coarse, cell_zero<Cell>()),
        F_right_f_cmp_(nj_coarse, cell_zero<Cell>()),
        G_bottom_c_(ni_coarse, cell_zero<Cell>()),
        G_top_c_(ni_coarse, cell_zero<Cell>()),
        G_bottom_c_cmp_(ni_coarse, cell_zero<Cell>()),
        G_top_c_cmp_(ni_coarse, cell_zero<Cell>()),
        G_bottom_f_sum_(ni_coarse, cell_zero<Cell>()),
        G_top_f_sum_(ni_coarse, cell_zero<Cell>()),
        G_bottom_f_cmp_(ni_coarse, cell_zero<Cell>()),
        G_top_f_cmp_(ni_coarse, cell_zero<Cell>()) {}

  void reset() noexcept {
    auto z = [](std::vector<Cell>& v) {
      std::fill(v.begin(), v.end(), cell_zero<Cell>());
    };
    z(F_left_c_); z(F_right_c_);
    z(F_left_c_cmp_); z(F_right_c_cmp_);
    z(F_left_f_sum_); z(F_right_f_sum_);
    z(F_left_f_cmp_); z(F_right_f_cmp_);
    z(G_bottom_c_); z(G_top_c_);
    z(G_bottom_c_cmp_); z(G_top_c_cmp_);
    z(G_bottom_f_sum_); z(G_top_f_sum_);
    z(G_bottom_f_cmp_); z(G_top_f_cmp_);
  }

  // Coarse side : overwrite (single value per coarse face per step). No
  // accumulation, no Kahan needed.
  void set_F_left_coarse(Index j_rel, const Cell& F) noexcept {
    F_left_c_[j_rel] = F;
  }
  void set_F_right_coarse(Index j_rel, const Cell& F) noexcept {
    F_right_c_[j_rel] = F;
  }
  void set_G_bottom_coarse(Index i_rel, const Cell& F) noexcept {
    G_bottom_c_[i_rel] = F;
  }
  void set_G_top_coarse(Index i_rel, const Cell& F) noexcept {
    G_top_c_[i_rel] = F;
  }

  // Coarse side : weighted accumulation. Used by multi-stage integrators
  // like SSPRK3 where the effective full-step flux is a weighted sum of
  // per-substage fluxes (weights 1/6, 1/6, 4/6 for Shu-Osher SSPRK3).
  // Caller must reset() before the first substage. Kahan-compensated for
  // the same FP-associativity reason as the fine side.
  void add_F_left_coarse_w(Index k, const Cell& F, Real w) noexcept {
    const Cell wF = w * F;
    kahan_add(F_left_c_[k], F_left_c_cmp_[k], wF);
  }
  void add_F_right_coarse_w(Index k, const Cell& F, Real w) noexcept {
    const Cell wF = w * F;
    kahan_add(F_right_c_[k], F_right_c_cmp_[k], wF);
  }
  void add_G_bottom_coarse_w(Index i, const Cell& F, Real w) noexcept {
    const Cell wF = w * F;
    kahan_add(G_bottom_c_[i], G_bottom_c_cmp_[i], wF);
  }
  void add_G_top_coarse_w(Index i, const Cell& F, Real w) noexcept {
    const Cell wF = w * F;
    kahan_add(G_top_c_[i], G_top_c_cmp_[i], wF);
  }

  // Fine side : weighted variant (SSPRK3 on the fine substages).
  void add_F_left_fine_w(Index k, const Cell& F, Real w) noexcept {
    const Cell wF = w * F;
    kahan_add(F_left_f_sum_[k], F_left_f_cmp_[k], wF);
  }
  void add_F_right_fine_w(Index k, const Cell& F, Real w) noexcept {
    const Cell wF = w * F;
    kahan_add(F_right_f_sum_[k], F_right_f_cmp_[k], wF);
  }
  void add_G_bottom_fine_w(Index k, const Cell& F, Real w) noexcept {
    const Cell wF = w * F;
    kahan_add(G_bottom_f_sum_[k], G_bottom_f_cmp_[k], wF);
  }
  void add_G_top_fine_w(Index k, const Cell& F, Real w) noexcept {
    const Cell wF = w * F;
    kahan_add(G_top_f_sum_[k], G_top_f_cmp_[k], wF);
  }

  // Fine side : unweighted accumulation, used by the plain
  // AMRIntegrator2D that does not substage-weight.
  void add_F_left_fine(Index k, const Cell& F) noexcept {
    kahan_add(F_left_f_sum_[k], F_left_f_cmp_[k], F);
  }
  void add_F_right_fine(Index k, const Cell& F) noexcept {
    kahan_add(F_right_f_sum_[k], F_right_f_cmp_[k], F);
  }
  void add_G_bottom_fine(Index k, const Cell& F) noexcept {
    kahan_add(G_bottom_f_sum_[k], G_bottom_f_cmp_[k], F);
  }
  void add_G_top_fine(Index k, const Cell& F) noexcept {
    kahan_add(G_top_f_sum_[k], G_top_f_cmp_[k], F);
  }

  // Per-thread fine-side accumulators. Reserve once before the parallel
  // for, write into [tid][k] slots without locking, merge after the
  // parallel region. The merge is Kahan-compensated and O(n_threads *
  // perimeter), which is tiny compared to the O(nx * ny) interior work.
  void reserve_threads(int n_threads) {
    if (n_threads <= n_threads_) return;
    n_threads_ = n_threads;
    auto resize_zero = [&](std::vector<std::vector<Cell>>& v,
                            std::size_t inner) {
      v.assign(n_threads_,
               std::vector<Cell>(inner, cell_zero<Cell>()));
    };
    resize_zero(tF_left_f_sum_,   nj_);
    resize_zero(tF_right_f_sum_,  nj_);
    resize_zero(tF_left_f_cmp_,   nj_);
    resize_zero(tF_right_f_cmp_,  nj_);
    resize_zero(tG_bottom_f_sum_, ni_);
    resize_zero(tG_top_f_sum_,    ni_);
    resize_zero(tG_bottom_f_cmp_, ni_);
    resize_zero(tG_top_f_cmp_,    ni_);
  }

  void add_F_left_fine_w_thread(Index k, const Cell& F, Real w, int tid) noexcept {
    const Cell wF = w * F;
    kahan_add(tF_left_f_sum_[tid][k], tF_left_f_cmp_[tid][k], wF);
  }
  void add_F_right_fine_w_thread(Index k, const Cell& F, Real w, int tid) noexcept {
    const Cell wF = w * F;
    kahan_add(tF_right_f_sum_[tid][k], tF_right_f_cmp_[tid][k], wF);
  }
  void add_G_bottom_fine_w_thread(Index k, const Cell& F, Real w, int tid) noexcept {
    const Cell wF = w * F;
    kahan_add(tG_bottom_f_sum_[tid][k], tG_bottom_f_cmp_[tid][k], wF);
  }
  void add_G_top_fine_w_thread(Index k, const Cell& F, Real w, int tid) noexcept {
    const Cell wF = w * F;
    kahan_add(tG_top_f_sum_[tid][k], tG_top_f_cmp_[tid][k], wF);
  }

  void add_F_left_fine_thread(Index k, const Cell& F, int tid) noexcept {
    kahan_add(tF_left_f_sum_[tid][k], tF_left_f_cmp_[tid][k], F);
  }
  void add_F_right_fine_thread(Index k, const Cell& F, int tid) noexcept {
    kahan_add(tF_right_f_sum_[tid][k], tF_right_f_cmp_[tid][k], F);
  }
  void add_G_bottom_fine_thread(Index k, const Cell& F, int tid) noexcept {
    kahan_add(tG_bottom_f_sum_[tid][k], tG_bottom_f_cmp_[tid][k], F);
  }
  void add_G_top_fine_thread(Index k, const Cell& F, int tid) noexcept {
    kahan_add(tG_top_f_sum_[tid][k], tG_top_f_cmp_[tid][k], F);
  }

  // Sequential Kahan merge of the per-thread slices into the canonical
  // accumulators, then zero out the per-thread storage so the next
  // substep starts fresh.
  void merge_thread_accumulators() noexcept {
    auto merge = [&](std::vector<std::vector<Cell>>& tsum,
                     std::vector<std::vector<Cell>>& tcmp,
                     std::vector<Cell>& sum,
                     std::vector<Cell>& cmp,
                     std::size_t inner) {
      for (int t = 0; t < n_threads_; ++t) {
        for (std::size_t k = 0; k < inner; ++k) {
          kahan_add(sum[k], cmp[k], tsum[t][k]);
          tsum[t][k] = cell_zero<Cell>();
          tcmp[t][k] = cell_zero<Cell>();
        }
      }
    };
    merge(tF_left_f_sum_,   tF_left_f_cmp_,   F_left_f_sum_,   F_left_f_cmp_,   nj_);
    merge(tF_right_f_sum_,  tF_right_f_cmp_,  F_right_f_sum_,  F_right_f_cmp_,  nj_);
    merge(tG_bottom_f_sum_, tG_bottom_f_cmp_, G_bottom_f_sum_, G_bottom_f_cmp_, ni_);
    merge(tG_top_f_sum_,    tG_top_f_cmp_,    G_top_f_sum_,    G_top_f_cmp_,    ni_);
  }

  // Reflux readouts
  const Cell& F_left_coarse(Index j_rel) const noexcept { return F_left_c_[j_rel]; }
  const Cell& F_right_coarse(Index j_rel) const noexcept { return F_right_c_[j_rel]; }
  const Cell& G_bottom_coarse(Index i_rel) const noexcept { return G_bottom_c_[i_rel]; }
  const Cell& G_top_coarse(Index i_rel) const noexcept { return G_top_c_[i_rel]; }

  Cell F_left_fine_avg(Index j_rel, Real r2_inv) const noexcept {
    return F_left_f_sum_[j_rel] * r2_inv;
  }
  Cell F_right_fine_avg(Index j_rel, Real r2_inv) const noexcept {
    return F_right_f_sum_[j_rel] * r2_inv;
  }
  Cell G_bottom_fine_avg(Index i_rel, Real r2_inv) const noexcept {
    return G_bottom_f_sum_[i_rel] * r2_inv;
  }
  Cell G_top_fine_avg(Index i_rel, Real r2_inv) const noexcept {
    return G_top_f_sum_[i_rel] * r2_inv;
  }

  Index ni() const noexcept { return ni_; }
  Index nj() const noexcept { return nj_; }

 private:
  Index ni_, nj_;
  std::vector<Cell> F_left_c_, F_right_c_;
  std::vector<Cell> F_left_c_cmp_, F_right_c_cmp_;
  std::vector<Cell> F_left_f_sum_, F_right_f_sum_;
  std::vector<Cell> F_left_f_cmp_, F_right_f_cmp_;
  std::vector<Cell> G_bottom_c_, G_top_c_;
  std::vector<Cell> G_bottom_c_cmp_, G_top_c_cmp_;
  std::vector<Cell> G_bottom_f_sum_, G_top_f_sum_;
  std::vector<Cell> G_bottom_f_cmp_, G_top_f_cmp_;

  int n_threads_ = 0;
  std::vector<std::vector<Cell>> tF_left_f_sum_, tF_right_f_sum_;
  std::vector<std::vector<Cell>> tF_left_f_cmp_, tF_right_f_cmp_;
  std::vector<std::vector<Cell>> tG_bottom_f_sum_, tG_top_f_sum_;
  std::vector<std::vector<Cell>> tG_bottom_f_cmp_, tG_top_f_cmp_;
};

}  // namespace pde_core
