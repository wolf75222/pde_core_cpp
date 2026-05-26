#pragma once

#include <pde_core/amr/mesh_hierarchy1d.hpp>
#include <pde_core/core/types.hpp>
#include <pde_core/mesh/field1d.hpp>

#include <algorithm>
#include <cstddef>

namespace pde_core {

// Re-fit the fine patch around regions flagged by `criterion`. Steps :
//   1. mark coarse cells via the criterion
//   2. compute new bounds [i_lo_new, i_hi_new) = bbox(marked) +/- buffer
//   3. if no cells marked, drop the fine block (return true if state changed)
//   4. if bounds unchanged, return false (no work)
//   5. allocate a new fine block and fill it from :
//        - the old fine block where the new patch overlaps the old one,
//        - parent coarse cells (piecewise-constant injection) elsewhere.
// PCI from coarse preserves discrete conservation exactly because the
// average of the r fine sub-cells equals the parent coarse value (after
// the average-down step that runs after every integrator step).
//
// `Criterion` is a callable `bool[] (const Field1D<Cell>&, Real dx)` that
// returns a per-cell mark vector.
template <class Cell, class Criterion>
bool regrid_1d(MeshHierarchy1D<Cell>& mesh, const Criterion& criterion,
               std::ptrdiff_t buffer = 4) {
  const auto& coarse = mesh.coarse().field();
  const Real dx_c = mesh.dx(0);
  const auto nx_c = static_cast<std::ptrdiff_t>(coarse.nx());
  const auto r = static_cast<std::ptrdiff_t>(MeshHierarchy1D<Cell>::refinement_ratio);
  const auto marks = criterion(coarse, dx_c);

  std::ptrdiff_t i_min = -1, i_max = -1;
  for (std::size_t i = 0; i < marks.size(); ++i) {
    if (marks[i]) {
      if (i_min < 0) i_min = static_cast<std::ptrdiff_t>(i);
      i_max = static_cast<std::ptrdiff_t>(i);
    }
  }

  if (i_min < 0) {
    if (mesh.has_fine()) {
      mesh.clear_refinement();
      return true;
    }
    return false;
  }

  // Expand by buffer, clamp, round to refinement-ratio alignment.
  i_min = std::max<std::ptrdiff_t>(0, i_min - buffer);
  i_max = std::min<std::ptrdiff_t>(nx_c - 1, i_max + buffer);
  std::ptrdiff_t i_hi_new = i_max + 1;
  std::ptrdiff_t i_lo_new = i_min;
  i_lo_new = (i_lo_new / r) * r;
  i_hi_new = ((i_hi_new + r - 1) / r) * r;
  i_hi_new = std::min<std::ptrdiff_t>(nx_c, i_hi_new);

  if (mesh.has_fine() && mesh.fine_lo_coarse() == i_lo_new &&
      mesh.fine_hi_coarse() == i_hi_new) {
    return false;
  }

  // Snapshot the old fine block (if any) before refine_region destroys it.
  Field1D<Cell> old_fine_copy(Index{1});
  const bool had_old = mesh.has_fine();
  std::ptrdiff_t old_lo = 0, old_hi = 0;
  if (had_old) {
    old_fine_copy = mesh.fine().field();
    old_lo = mesh.fine_lo_coarse();
    old_hi = mesh.fine_hi_coarse();
  }

  mesh.refine_region(i_lo_new, i_hi_new);
  auto& new_fine = mesh.fine().field();
  const auto nxf = static_cast<std::ptrdiff_t>(new_fine.nx());
  const std::ptrdiff_t new_base_global = i_lo_new * r;
  const std::ptrdiff_t old_base_global = old_lo * r;

  for (std::ptrdiff_t k = 0; k < nxf; ++k) {
    const std::ptrdiff_t k_global = new_base_global + k;
    const std::ptrdiff_t k_coarse = k_global / r;
    if (had_old && k_global >= old_lo * r && k_global < old_hi * r) {
      new_fine(k) = old_fine_copy(k_global - old_base_global);
    } else {
      new_fine(k) = coarse(k_coarse);
    }
  }
  return true;
}

}  // namespace pde_core
