#pragma once

#include <pde_core/amr/box2d.hpp>
#include <pde_core/amr/mesh_block2d.hpp>
#include <pde_core/core/types.hpp>

#include <cmath>
#include <cstddef>
#include <deque>
#include <optional>
#include <vector>

namespace pde_core {

// 2D mesh hierarchy with refinement ratio 2. Templated on the per-cell
// payload Cell. Supports three access patterns layered on top of each
// other ; all three can be mixed within the same hierarchy and remain
// bit-equivalent in the single-patch case :
//
//  1. **Legacy two-level API**. A single fine block at level 1.
//     Methods : has_fine(), fine(), fine_lo_*_coarse().
//  2. **Single-patch multi-level API**. Up to max_level (= 8) nested fine
//     blocks, one block per level. Methods : is_refined(L), block(L),
//     coverage(L), refine_at_level(L, ...), finest_level().
//  3. **Multi-patch per-level API**. Each level may hold an arbitrary
//     number of disjoint patches. Methods : n_patches(L), patch(L, k),
//     patch_coverage(L, k), set_patches_at_level(L, [covs...]),
//     add_patch_at_level(L, cov). Patches at a given level are stored in
//     a std::deque so that adding patches does not invalidate references
//     to existing patches.
//
// Backward-compat invariants : block(L) is the same as patch(L, 0), and
// coverage(L) is the same as patch_coverage(L, 0). When at most one
// patch per level exists, every legacy code path stays bit-equivalent.
template <class Cell>
class MeshHierarchy2D {
 public:
  static constexpr int max_level = 8;
  static constexpr Index refinement_ratio = 2;

  // Coverage of a fine patch in its parent (L-1) index space. The parent
  // here is the canonical, global-domain-aligned level L-1 grid, not a
  // particular parent patch.
  struct Coverage {
    std::ptrdiff_t i_lo = 0, i_hi = 0;
    std::ptrdiff_t j_lo = 0, j_hi = 0;
  };

  MeshHierarchy2D(Real x_min, Real x_max, Real y_min, Real y_max,
                  Index nx_base, Index ny_base, Index ghost = 2)
      : x_min_(x_min), x_max_(x_max),
        y_min_(y_min), y_max_(y_max),
        nx_base_(nx_base), ny_base_(ny_base),
        ghost_(ghost),
        coarse_(Box2D{0, static_cast<std::ptrdiff_t>(nx_base),
                       0, static_cast<std::ptrdiff_t>(ny_base), 0}, ghost) {}

  // ---- single-patch refinement (level-1 only) -------------------------

  // Legacy two-level entry : level-1 single patch over the given coarse
  // cell range. Equivalent to refine_at_level(1, ...).
  void refine_region(std::ptrdiff_t i_lo_coarse, std::ptrdiff_t i_hi_coarse,
                     std::ptrdiff_t j_lo_coarse, std::ptrdiff_t j_hi_coarse) {
    refine_at_level(1, i_lo_coarse, i_hi_coarse, j_lo_coarse, j_hi_coarse);
  }

  // Set level L to a SINGLE patch covering [i_lo, i_hi) x [j_lo, j_hi)
  // in the parent's local cell space. Replaces any existing patches at
  // level L and wipes all deeper levels (they become orphan).
  // Pre : 1 <= L <= max_level and level L-1 must already exist.
  void refine_at_level(int L,
                        std::ptrdiff_t i_lo, std::ptrdiff_t i_hi,
                        std::ptrdiff_t j_lo, std::ptrdiff_t j_hi) {
    set_patches_at_level(L, {Coverage{i_lo, i_hi, j_lo, j_hi}});
  }

  // ---- multi-patch refinement -----------------------------------------

  // Replace all patches at level L with the given list. Drops any deeper
  // levels. Pre : 1 <= L <= max_level, level L-1 must exist, each
  // coverage is in L-1's local cell space.
  void set_patches_at_level(int L, std::vector<Coverage> covs) {
    if (L < 1 || L > max_level) return;
    if (L > 1 && !is_refined(L - 1)) return;
    if (static_cast<int>(patches_.size()) < L) {
      patches_.resize(L);
      coverages_per_patch_.resize(L);
    }
    if (static_cast<int>(patches_.size()) > L) {
      patches_.resize(L);
      coverages_per_patch_.resize(L);
    }
    patches_[L - 1].clear();
    coverages_per_patch_[L - 1].clear();
    for (const auto& cov : covs) {
      add_patch_internal(L, cov);
    }
    update_legacy_mirror(L);
  }

  // Append one patch to level L (does NOT replace existing patches).
  // Pre : 1 <= L <= max_level, level L-1 must exist. The new patch's
  // coverage is in L-1's local cell space and should not overlap with
  // existing patches at the same level (the integrator assumes patches
  // are disjoint).
  void add_patch_at_level(int L, const Coverage& cov) {
    if (L < 1 || L > max_level) return;
    if (L > 1 && !is_refined(L - 1)) return;
    if (static_cast<int>(patches_.size()) < L) {
      patches_.resize(L);
      coverages_per_patch_.resize(L);
    }
    if (static_cast<int>(patches_.size()) > L) {
      patches_.resize(L);
      coverages_per_patch_.resize(L);
    }
    add_patch_internal(L, cov);
    update_legacy_mirror(L);
  }

  void clear_refinement() noexcept {
    fine_present_ = false;
    patches_.clear();
    coverages_per_patch_.clear();
  }

  // ---- geometry --------------------------------------------------------
  Real x_min() const noexcept { return x_min_; }
  Real x_max() const noexcept { return x_max_; }
  Real y_min() const noexcept { return y_min_; }
  Real y_max() const noexcept { return y_max_; }
  Index nx_base() const noexcept { return nx_base_; }
  Index ny_base() const noexcept { return ny_base_; }
  Index ghost() const noexcept { return ghost_; }

  Real dx(int level) const noexcept {
    const Real base = (x_max_ - x_min_) / static_cast<Real>(nx_base_);
    return base / std::pow(static_cast<Real>(refinement_ratio), level);
  }
  Real dy(int level) const noexcept {
    const Real base = (y_max_ - y_min_) / static_cast<Real>(ny_base_);
    return base / std::pow(static_cast<Real>(refinement_ratio), level);
  }
  Real x_cell(int level, std::ptrdiff_t i) const noexcept {
    return x_min_ + (static_cast<Real>(i) + Real{0.5}) * dx(level);
  }
  Real y_cell(int level, std::ptrdiff_t j) const noexcept {
    return y_min_ + (static_cast<Real>(j) + Real{0.5}) * dy(level);
  }

  // ---- legacy two-level API (unchanged behaviour) ---------------------
  MeshBlock2D<Cell>& coarse() noexcept { return coarse_; }
  const MeshBlock2D<Cell>& coarse() const noexcept { return coarse_; }
  bool has_fine() const noexcept { return fine_present_; }
  MeshBlock2D<Cell>& fine() noexcept { return patches_[0].front(); }
  const MeshBlock2D<Cell>& fine() const noexcept { return patches_[0].front(); }

  std::ptrdiff_t fine_lo_i_coarse() const noexcept { return fine_lo_i_coarse_; }
  std::ptrdiff_t fine_hi_i_coarse() const noexcept { return fine_hi_i_coarse_; }
  std::ptrdiff_t fine_lo_j_coarse() const noexcept { return fine_lo_j_coarse_; }
  std::ptrdiff_t fine_hi_j_coarse() const noexcept { return fine_hi_j_coarse_; }

  // ---- single-patch multi-level API -----------------------------------
  // (returns the FIRST patch at level L.)

  bool is_refined(int L) const noexcept {
    if (L == 0) return true;
    if (L < 1 || L > static_cast<int>(patches_.size())) return false;
    return !patches_[L - 1].empty();
  }

  int finest_level() const noexcept {
    int L = 0;
    for (std::size_t k = 0; k < patches_.size(); ++k)
      if (!patches_[k].empty()) L = static_cast<int>(k + 1);
    return L;
  }

  MeshBlock2D<Cell>& block(int L) noexcept {
    return (L == 0) ? coarse_ : patches_[L - 1].front();
  }
  const MeshBlock2D<Cell>& block(int L) const noexcept {
    return (L == 0) ? coarse_ : patches_[L - 1].front();
  }
  const Coverage& coverage(int L) const noexcept {
    return coverages_per_patch_[L - 1].front();
  }

  // ---- multi-patch API ------------------------------------------------

  // Number of patches at level L. Level 0 always has 1 patch (the coarse
  // base). Levels with no refinement have 0 patches.
  int n_patches(int L) const noexcept {
    if (L == 0) return 1;
    if (L < 1 || L > static_cast<int>(patches_.size())) return 0;
    return static_cast<int>(patches_[L - 1].size());
  }

  // k-th patch at level L. Level 0 is always the coarse block.
  MeshBlock2D<Cell>& patch(int L, int k) noexcept {
    return (L == 0) ? coarse_ : patches_[L - 1][static_cast<std::size_t>(k)];
  }
  const MeshBlock2D<Cell>& patch(int L, int k) const noexcept {
    return (L == 0) ? coarse_ : patches_[L - 1][static_cast<std::size_t>(k)];
  }
  const Coverage& patch_coverage(int L, int k) const noexcept {
    return coverages_per_patch_[L - 1][static_cast<std::size_t>(k)];
  }

 private:
  // Allocate a fresh MeshBlock2D<Cell> for the given patch at level L and
  // append it to the per-level patch deque. Deque keeps references stable
  // across push_back, unlike std::vector.
  void add_patch_internal(int L, const Coverage& cov) {
    const auto r = static_cast<std::ptrdiff_t>(refinement_ratio);
    // Box indices are global (level-L cell offset from x_min) so that
    // x_cell(L, box.i_lo + i_local) gives the correct physical position.
    // For L = 1, parent (L0) has box.i_lo = 0, so this is r * cov.i_lo.
    // For L >= 2, the cov is interpreted in the parent's local cell
    // space. We anchor against patches_[L-2].front() (the first parent
    // patch) ; with multi-patch hierarchies, all parent patches share
    // the same canonical L-1 cell grid so this anchor is well defined.
    std::ptrdiff_t parent_i_lo = 0;
    std::ptrdiff_t parent_j_lo = 0;
    if (L >= 2) {
      parent_i_lo = patches_[L - 2].front().box().i_lo;
      parent_j_lo = patches_[L - 2].front().box().j_lo;
    }
    patches_[L - 1].emplace_back(
        Box2D{r * (parent_i_lo + cov.i_lo), r * (parent_i_lo + cov.i_hi),
              r * (parent_j_lo + cov.j_lo), r * (parent_j_lo + cov.j_hi), L},
        ghost_);
    coverages_per_patch_[L - 1].push_back(cov);
  }

  void update_legacy_mirror(int L) noexcept {
    if (L != 1) return;
    if (patches_[0].empty()) {
      fine_present_ = false;
      return;
    }
    fine_present_ = true;
    const auto& cov = coverages_per_patch_[0].front();
    fine_lo_i_coarse_ = cov.i_lo;
    fine_hi_i_coarse_ = cov.i_hi;
    fine_lo_j_coarse_ = cov.j_lo;
    fine_hi_j_coarse_ = cov.j_hi;
  }

  Real x_min_, x_max_, y_min_, y_max_;
  Index nx_base_, ny_base_, ghost_;
  MeshBlock2D<Cell> coarse_;

  // Legacy two-level metadata mirroring patches_[0].front().
  bool fine_present_ = false;
  std::ptrdiff_t fine_lo_i_coarse_ = 0;
  std::ptrdiff_t fine_hi_i_coarse_ = 0;
  std::ptrdiff_t fine_lo_j_coarse_ = 0;
  std::ptrdiff_t fine_hi_j_coarse_ = 0;

  // Multi-patch storage. patches_[L-1] is a deque (push_back does not
  // invalidate references) of all patches at level L. coverages_per_patch_
  // mirrors it with each patch's coverage in parent local cell space.
  std::vector<std::deque<MeshBlock2D<Cell>>> patches_;
  std::vector<std::vector<Coverage>> coverages_per_patch_;
};

}  // namespace pde_core
