#pragma once

#include <pde_core/amr/mesh_hierarchy2d.hpp>
#include <pde_core/core/openmp.hpp>
#include <pde_core/core/types.hpp>
#include <pde_core/mesh/field2d.hpp>

#include <cstddef>

namespace pde_core {

// Per-patch ghost fill for multi-patch AMR.
//
// For each ghost cell of patch (L, k), the source is picked in this order :
//   1. Default : parent (level L-1) cell with linear-in-time interpolation
//      between the snapshot parent_n (start of coarse step) and parent_np1
//      (after the coarse step), weighted by alpha in [0, 1]. The caller is
//      responsible for having applied the boundary conditions to parent_n
//      before snapshotting it ; this is how patches that touch the global
//      domain boundary get BC-respecting ghosts.
//   2. Sibling override : if the ghost cell falls inside another L-level
//      patch's INTERIOR, the parent value is overwritten by the sibling's
//      cell value. No time interpolation since the sibling is at the same
//      time level as the fine patch, and the sibling data is more accurate
//      than the piecewise-injected parent.
//
// Pre : the canonical level L-1 grid is single-patch (mesh.block(L - 1)
// for L >= 2, or the coarse base for L = 1). Fully multi-patch parents
// would require an extra dispatch over the parent's patches at fill time.
template <class Cell>
inline void fill_patch_ghosts_multipatch(
    const MeshHierarchy2D<Cell>& mesh, int L, int patch_idx,
    const Field2D<Cell>& parent_n, const Field2D<Cell>& parent_np1,
    Real alpha) {
  auto& patch_blk = const_cast<MeshHierarchy2D<Cell>&>(mesh).patch(L, patch_idx);
  const auto g = static_cast<std::ptrdiff_t>(patch_blk.field().ghost());
  const auto r = static_cast<std::ptrdiff_t>(
      MeshHierarchy2D<Cell>::refinement_ratio);
  const auto patch_box = patch_blk.box();
  const auto nxf = static_cast<std::ptrdiff_t>(patch_blk.nx());
  const auto nyf = static_cast<std::ptrdiff_t>(patch_blk.ny());

  // For L = 1 the parent is the coarse base (box.i_lo = 0). For L >= 2 the
  // parent is mesh.block(L - 1) and its box may carry an offset.
  std::ptrdiff_t parent_box_i_lo = 0;
  std::ptrdiff_t parent_box_j_lo = 0;
  if (L >= 2) {
    parent_box_i_lo = mesh.block(L - 1).box().i_lo;
    parent_box_j_lo = mesh.block(L - 1).box().j_lo;
  }

  auto floor_div = [r](std::ptrdiff_t a) {
    return a >= 0 ? a / r : -((-a + r - 1) / r);
  };

  auto fill_from_parent = [&](std::ptrdiff_t i_local, std::ptrdiff_t j_local) {
    const auto gi_L = patch_box.i_lo + i_local;
    const auto gj_L = patch_box.j_lo + j_local;
    const auto ip_global = floor_div(gi_L);
    const auto jp_global = floor_div(gj_L);
    const auto ip = ip_global - parent_box_i_lo;
    const auto jp = jp_global - parent_box_j_lo;
    patch_blk.field()(i_local, j_local) =
        (Real{1} - alpha) * parent_n(ip, jp) + alpha * parent_np1(ip, jp);
  };

  auto try_fill_from_sibling = [&](std::ptrdiff_t i_local,
                                    std::ptrdiff_t j_local) -> bool {
    const auto gi_L = patch_box.i_lo + i_local;
    const auto gj_L = patch_box.j_lo + j_local;
    const int n = mesh.n_patches(L);
    for (int k = 0; k < n; ++k) {
      if (k == patch_idx) continue;
      const auto& sib_blk = mesh.patch(L, k);
      const auto sb = sib_blk.box();
      if (gi_L >= sb.i_lo && gi_L < sb.i_hi &&
          gj_L >= sb.j_lo && gj_L < sb.j_hi) {
        patch_blk.field()(i_local, j_local) =
            sib_blk.field()(gi_L - sb.i_lo, gj_L - sb.j_lo);
        return true;
      }
    }
    return false;
  };

  // Pass 1 : left and right ghost columns, full vertical span (including
  // y ghost cells) so corners are written exactly once. Iterations across
  // j touch independent cells, so the parallel for is race-free. The
  // outer k loop also iterates over disjoint cells.
  for (std::ptrdiff_t k = 1; k <= g; ++k) {
    PDE_OMP_PARALLEL_FOR
    for (std::ptrdiff_t j = -g; j < nyf + g; ++j) {
      fill_from_parent(-k, j);
      try_fill_from_sibling(-k, j);
      fill_from_parent(nxf - 1 + k, j);
      try_fill_from_sibling(nxf - 1 + k, j);
    }
  }
  // Pass 2 : bottom and top ghost rows on interior columns only, corners
  // already done by pass 1.
  for (std::ptrdiff_t k = 1; k <= g; ++k) {
    PDE_OMP_PARALLEL_FOR
    for (std::ptrdiff_t i = 0; i < nxf; ++i) {
      fill_from_parent(i, -k);
      try_fill_from_sibling(i, -k);
      fill_from_parent(i, nyf - 1 + k);
      try_fill_from_sibling(i, nyf - 1 + k);
    }
  }
}

}  // namespace pde_core
