#pragma once

#include <pde_core/amr/clustering2d.hpp>
#include <pde_core/amr/mesh_hierarchy2d.hpp>
#include <pde_core/core/types.hpp>
#include <pde_core/mesh/field2d.hpp>

#include <algorithm>
#include <cstddef>

namespace pde_core {

// 2D single-block regrid. Pipeline :
//   1. apply `criterion` to the coarse cells, produce a row-major boolean
//      mark grid of size nx*ny
//   2. find the bounding box of the marked cells in both directions
//   3. if none marked, drop the fine block and report a state change
//   4. expand bbox by `buffer`, clamp to the domain, round each axis to
//      refinement-ratio alignment
//   5. if the bounds are unchanged, return false (no work)
//   6. snapshot the old fine block, allocate the new fine block, fill it
//      from :
//        - the old fine block on the overlap (preserves high-resolution
//          data),
//        - parent coarse cells elsewhere via piecewise-constant injection.
template <class Cell, class Criterion>
bool regrid_2d(MeshHierarchy2D<Cell>& mesh, const Criterion& criterion,
               std::ptrdiff_t buffer = 4) {
  const auto& coarse = mesh.coarse().field();
  const Real dx_c = mesh.dx(0);
  const Real dy_c = mesh.dy(0);
  const auto nx_c = static_cast<std::ptrdiff_t>(coarse.nx());
  const auto ny_c = static_cast<std::ptrdiff_t>(coarse.ny());
  const auto r = static_cast<std::ptrdiff_t>(MeshHierarchy2D<Cell>::refinement_ratio);
  const auto marks = criterion(coarse, dx_c, dy_c);

  std::ptrdiff_t i_min = -1, i_max = -1, j_min = -1, j_max = -1;
  for (std::ptrdiff_t j = 0; j < ny_c; ++j) {
    for (std::ptrdiff_t i = 0; i < nx_c; ++i) {
      if (marks[static_cast<std::size_t>(j * nx_c + i)]) {
        if (i_min < 0) {
          i_min = i_max = i;
          j_min = j_max = j;
        } else {
          i_min = std::min(i_min, i);
          i_max = std::max(i_max, i);
          j_min = std::min(j_min, j);
          j_max = std::max(j_max, j);
        }
      }
    }
  }

  if (i_min < 0) {
    if (mesh.has_fine()) {
      mesh.clear_refinement();
      return true;
    }
    return false;
  }

  i_min = std::max<std::ptrdiff_t>(0, i_min - buffer);
  i_max = std::min<std::ptrdiff_t>(nx_c - 1, i_max + buffer);
  j_min = std::max<std::ptrdiff_t>(0, j_min - buffer);
  j_max = std::min<std::ptrdiff_t>(ny_c - 1, j_max + buffer);
  std::ptrdiff_t i_lo_new = (i_min / r) * r;
  std::ptrdiff_t i_hi_new = std::min<std::ptrdiff_t>(
      nx_c, ((i_max + 1 + r - 1) / r) * r);
  std::ptrdiff_t j_lo_new = (j_min / r) * r;
  std::ptrdiff_t j_hi_new = std::min<std::ptrdiff_t>(
      ny_c, ((j_max + 1 + r - 1) / r) * r);

  if (mesh.has_fine() && mesh.fine_lo_i_coarse() == i_lo_new &&
      mesh.fine_hi_i_coarse() == i_hi_new &&
      mesh.fine_lo_j_coarse() == j_lo_new &&
      mesh.fine_hi_j_coarse() == j_hi_new) {
    return false;
  }

  Field2D<Cell> old_copy(Index{1}, Index{1});
  const bool had_old = mesh.has_fine();
  std::ptrdiff_t old_i_lo = 0, old_i_hi = 0, old_j_lo = 0, old_j_hi = 0;
  if (had_old) {
    old_copy = mesh.fine().field();
    old_i_lo = mesh.fine_lo_i_coarse();
    old_i_hi = mesh.fine_hi_i_coarse();
    old_j_lo = mesh.fine_lo_j_coarse();
    old_j_hi = mesh.fine_hi_j_coarse();
  }

  mesh.refine_region(i_lo_new, i_hi_new, j_lo_new, j_hi_new);
  auto& new_fine = mesh.fine().field();
  const auto nxf = static_cast<std::ptrdiff_t>(new_fine.nx());
  const auto nyf = static_cast<std::ptrdiff_t>(new_fine.ny());
  const auto new_base_i = i_lo_new * r;
  const auto new_base_j = j_lo_new * r;
  const auto old_base_i = old_i_lo * r;
  const auto old_base_j = old_j_lo * r;
  const auto old_top_i = old_i_hi * r;
  const auto old_top_j = old_j_hi * r;

  for (std::ptrdiff_t jf = 0; jf < nyf; ++jf) {
    for (std::ptrdiff_t ifn = 0; ifn < nxf; ++ifn) {
      const auto ig = new_base_i + ifn;
      const auto jg = new_base_j + jf;
      const bool in_old = had_old && ig >= old_base_i && ig < old_top_i &&
                          jg >= old_base_j && jg < old_top_j;
      if (in_old) {
        new_fine(ifn, jf) = old_copy(ig - old_base_i, jg - old_base_j);
      } else {
        new_fine(ifn, jf) = coarse(ig / r, jg / r);
      }
    }
  }
  return true;
}

// Multi-level regrid : one block per level, ratio 2, up to max_target_level
// (clamped to MeshHierarchy2D::max_level). The hierarchy is rebuilt
// bottom-up :
//   level 0 = the coarse base
//   level 1 = bbox of cells marked by criterion(coarse field)
//   level 2 = bbox of L1 cells marked by criterion(L1 field), clipped to
//             cov(1) with a `nest_buffer` cells margin (Berger-Colella
//             proper nesting : L+1 sits strictly inside L with at least
//             `nest_buffer` cells of L separation from L's edge)
//   ...
// Returns true if the hierarchy structure changed. Each new level's data
// is initialised by piecewise-constant injection from its parent.
template <class Cell, class Criterion>
bool regrid_2d_multilevel(MeshHierarchy2D<Cell>& mesh,
                           const Criterion& criterion,
                           int max_target_level = 2,
                           std::ptrdiff_t buffer = 4,
                           std::ptrdiff_t nest_buffer = 2) {
  if (max_target_level < 1) {
    if (mesh.finest_level() > 0) {
      mesh.clear_refinement();
      return true;
    }
    return false;
  }
  if (max_target_level > MeshHierarchy2D<Cell>::max_level) {
    max_target_level = MeshHierarchy2D<Cell>::max_level;
  }
  const auto r = static_cast<std::ptrdiff_t>(MeshHierarchy2D<Cell>::refinement_ratio);

  const int old_finest = mesh.finest_level();
  std::vector<typename MeshHierarchy2D<Cell>::Coverage> old_covs(old_finest);
  for (int L = 1; L <= old_finest; ++L) old_covs[L - 1] = mesh.coverage(L);

  // Tear down everything from L = 1 so refine_at_level rebuilds cleanly,
  // then add levels one at a time.
  mesh.clear_refinement();

  bool changed = (old_finest != 0);

  for (int L = 1; L <= max_target_level; ++L) {
    const auto& parent_blk = mesh.block(L - 1);
    const auto& parent = parent_blk.field();
    const Real dx_p = mesh.dx(L - 1);
    const Real dy_p = mesh.dy(L - 1);
    const auto nx_p = static_cast<std::ptrdiff_t>(parent.nx());
    const auto ny_p = static_cast<std::ptrdiff_t>(parent.ny());
    const auto marks = criterion(parent, dx_p, dy_p);

    // Restrict the search to the interior of the parent (avoid putting L
    // on the parent's perimeter ; the proper-nesting buffer is required
    // for L >= 2 because the L-(L-1) ghost fill assumes parent cells
    // exist one cell beyond cov(L)).
    const std::ptrdiff_t left_min  = (L == 1) ? 0 : nest_buffer;
    const std::ptrdiff_t right_max = (L == 1) ? nx_p - 1 : nx_p - 1 - nest_buffer;
    const std::ptrdiff_t bot_min   = (L == 1) ? 0 : nest_buffer;
    const std::ptrdiff_t top_max   = (L == 1) ? ny_p - 1 : ny_p - 1 - nest_buffer;

    std::ptrdiff_t i_min = -1, i_max = -1, j_min = -1, j_max = -1;
    for (std::ptrdiff_t j = bot_min; j <= top_max; ++j) {
      for (std::ptrdiff_t i = left_min; i <= right_max; ++i) {
        if (marks[static_cast<std::size_t>(j * nx_p + i)]) {
          if (i_min < 0) {
            i_min = i_max = i;
            j_min = j_max = j;
          } else {
            i_min = std::min(i_min, i);
            i_max = std::max(i_max, i);
            j_min = std::min(j_min, j);
            j_max = std::max(j_max, j);
          }
        }
      }
    }
    if (i_min < 0) {
      break;
    }
    i_min = std::max(left_min,  i_min - buffer);
    i_max = std::min(right_max, i_max + buffer);
    j_min = std::max(bot_min,   j_min - buffer);
    j_max = std::min(top_max,   j_max + buffer);
    const std::ptrdiff_t i_lo_new = (i_min / r) * r;
    const std::ptrdiff_t i_hi_new = std::min(nx_p, ((i_max + 1 + r - 1) / r) * r);
    const std::ptrdiff_t j_lo_new = (j_min / r) * r;
    const std::ptrdiff_t j_hi_new = std::min(ny_p, ((j_max + 1 + r - 1) / r) * r);
    // Copy parent's field before refine_at_level : the call resizes the
    // patches storage and can move the existing blocks, invalidating the
    // `parent` reference. The copy is small (one block of cells)
    // compared to the rebuild + injection cost.
    Field2D<Cell> parent_copy = parent;
    mesh.refine_at_level(L, i_lo_new, i_hi_new, j_lo_new, j_hi_new);

    auto& new_blk = mesh.block(L);
    auto& new_U = new_blk.field();
    const auto nxf = static_cast<std::ptrdiff_t>(new_blk.nx());
    const auto nyf = static_cast<std::ptrdiff_t>(new_blk.ny());
    for (std::ptrdiff_t jf = 0; jf < nyf; ++jf) {
      for (std::ptrdiff_t ifn = 0; ifn < nxf; ++ifn) {
        const auto ip = i_lo_new + ifn / r;
        const auto jp = j_lo_new + jf / r;
        new_U(ifn, jf) = parent_copy(ip, jp);
      }
    }

    if (L > old_finest) {
      changed = true;
    } else {
      const auto& ocov = old_covs[L - 1];
      if (ocov.i_lo != i_lo_new || ocov.i_hi != i_hi_new ||
          ocov.j_lo != j_lo_new || ocov.j_hi != j_hi_new) {
        changed = true;
      }
    }
  }

  if (mesh.finest_level() < old_finest) changed = true;

  return changed;
}

// Multi-patch regrid : run Berger-Rigoutsos clustering on the criterion
// marks and produce one or more patches at level 1, instead of the single
// bounding-box patch produced by `regrid_2d`.
//
// Pipeline :
//   1. apply `criterion` to the coarse field, produce boolean marks
//   2. cluster the marks via `cluster_berger_rigoutsos`, get a list of Box2D
//   3. expand each box by `buffer`, snap to refinement-ratio alignment,
//      clamp to the coarse domain
//   4. merge any patches that now overlap after buffering (rare ; happens
//      when two clusters are close enough that their buffered boxes touch)
//   5. set the patches at level 1 via `set_patches_at_level`
//   6. initialise each patch by piecewise-constant injection from the
//      coarse cell that covers it
// Returns true if the patch structure changed.
template <class Cell, class Criterion>
bool regrid_2d_multipatch(MeshHierarchy2D<Cell>& mesh,
                           const Criterion& criterion,
                           const ClusteringParams2D& cluster_params = {},
                           std::ptrdiff_t buffer = 4) {
  const auto& coarse = mesh.coarse().field();
  const Real dx_c = mesh.dx(0);
  const Real dy_c = mesh.dy(0);
  const auto nx_c = static_cast<std::ptrdiff_t>(coarse.nx());
  const auto ny_c = static_cast<std::ptrdiff_t>(coarse.ny());
  const auto r = static_cast<std::ptrdiff_t>(MeshHierarchy2D<Cell>::refinement_ratio);

  const auto marks = criterion(coarse, dx_c, dy_c);
  auto boxes = cluster_berger_rigoutsos(marks, nx_c, ny_c, cluster_params);

  const int old_n = mesh.n_patches(1);
  std::vector<typename MeshHierarchy2D<Cell>::Coverage> old_covs;
  old_covs.reserve(static_cast<std::size_t>(old_n));
  for (int k = 0; k < old_n; ++k) {
    old_covs.push_back(mesh.patch_coverage(1, k));
  }

  if (boxes.empty()) {
    if (old_n > 0) {
      mesh.clear_refinement();
      return true;
    }
    return false;
  }

  std::vector<typename MeshHierarchy2D<Cell>::Coverage> covs;
  covs.reserve(boxes.size());
  for (const auto& b : boxes) {
    std::ptrdiff_t i_lo = std::max<std::ptrdiff_t>(0,    b.i_lo - buffer);
    std::ptrdiff_t i_hi = std::min<std::ptrdiff_t>(nx_c, b.i_hi + buffer);
    std::ptrdiff_t j_lo = std::max<std::ptrdiff_t>(0,    b.j_lo - buffer);
    std::ptrdiff_t j_hi = std::min<std::ptrdiff_t>(ny_c, b.j_hi + buffer);
    i_lo = (i_lo / r) * r;
    i_hi = std::min(nx_c, ((i_hi + r - 1) / r) * r);
    j_lo = (j_lo / r) * r;
    j_hi = std::min(ny_c, ((j_hi + r - 1) / r) * r);
    if (i_hi <= i_lo || j_hi <= j_lo) continue;
    covs.push_back({i_lo, i_hi, j_lo, j_hi});
  }

  // Greedy merge of any patches that overlap after buffering.
  bool merged = true;
  while (merged && covs.size() >= 2) {
    merged = false;
    for (std::size_t a = 0; a + 1 < covs.size() && !merged; ++a) {
      for (std::size_t b = a + 1; b < covs.size(); ++b) {
        const auto& A = covs[a];
        const auto& B = covs[b];
        const bool overlap = !(A.i_hi <= B.i_lo || B.i_hi <= A.i_lo ||
                                A.j_hi <= B.j_lo || B.j_hi <= A.j_lo);
        if (overlap) {
          covs[a] = {std::min(A.i_lo, B.i_lo), std::max(A.i_hi, B.i_hi),
                      std::min(A.j_lo, B.j_lo), std::max(A.j_hi, B.j_hi)};
          covs.erase(covs.begin() + static_cast<std::ptrdiff_t>(b));
          merged = true;
          break;
        }
      }
    }
  }

  bool changed = (covs.size() != static_cast<std::size_t>(old_n));
  if (!changed) {
    for (std::size_t k = 0; k < covs.size(); ++k) {
      const auto& a = covs[k];
      const auto& b = old_covs[k];
      if (a.i_lo != b.i_lo || a.i_hi != b.i_hi ||
          a.j_lo != b.j_lo || a.j_hi != b.j_hi) {
        changed = true;
        break;
      }
    }
  }
  if (!changed) return false;

  // Copy coarse field before mutating the hierarchy (set_patches_at_level
  // may resize the patch storage and we want a stable source for the
  // PCM injection).
  Field2D<Cell> coarse_copy = coarse;
  mesh.set_patches_at_level(1, covs);

  for (int k = 0; k < mesh.n_patches(1); ++k) {
    const auto& cov = mesh.patch_coverage(1, k);
    auto& blk = mesh.patch(1, k);
    auto& U = blk.field();
    const auto nxf = static_cast<std::ptrdiff_t>(blk.nx());
    const auto nyf = static_cast<std::ptrdiff_t>(blk.ny());
    for (std::ptrdiff_t jf = 0; jf < nyf; ++jf) {
      for (std::ptrdiff_t ifn = 0; ifn < nxf; ++ifn) {
        const auto ip = cov.i_lo + ifn / r;
        const auto jp = cov.j_lo + jf / r;
        U(ifn, jf) = coarse_copy(ip, jp);
      }
    }
  }
  return true;
}

}  // namespace pde_core
