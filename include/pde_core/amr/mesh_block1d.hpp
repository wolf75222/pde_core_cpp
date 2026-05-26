#pragma once

#include <pde_core/amr/box1d.hpp>
#include <pde_core/mesh/field1d.hpp>

namespace pde_core {

// Self-contained unit of work a la Athena++ MeshBlock : a Box1D giving the
// index-space coverage, plus a Field1D<Cell> with ghost cells holding the
// solution. Different blocks may live at different refinement levels ; all
// blocks carry ghosts on both sides regardless of level.
//
// Cell is the per-cell payload :
//   - scalar problems : MeshBlock1D<Real>
//   - 1D Euler        : MeshBlock1D<ConservedState> via consumer alias
template <class Cell>
class MeshBlock1D {
 public:
  MeshBlock1D(Box1D box, Index ghost = 2)
      : box_(box), field_(box.nx(), ghost) {}

  const Box1D& box() const noexcept { return box_; }
  int level() const noexcept { return box_.level; }
  Index nx() const noexcept { return box_.nx(); }

  Field1D<Cell>& field() noexcept { return field_; }
  const Field1D<Cell>& field() const noexcept { return field_; }

 private:
  Box1D box_;
  Field1D<Cell> field_;
};

}  // namespace pde_core
