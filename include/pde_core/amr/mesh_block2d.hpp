#pragma once

#include <pde_core/amr/box2d.hpp>
#include <pde_core/mesh/field2d.hpp>

namespace pde_core {

// 2D MeshBlock : a Box2D giving the index-space coverage, plus a
// Field2D<Cell> with ghost cells holding the solution. Ghost count is the
// same on every level, set by the constructor (default 2).
template <class Cell>
class MeshBlock2D {
 public:
  MeshBlock2D(Box2D box, Index ghost = 2)
      : box_(box), field_(box.nx(), box.ny(), ghost) {}

  const Box2D& box() const noexcept { return box_; }
  int level() const noexcept { return box_.level; }
  Index nx() const noexcept { return box_.nx(); }
  Index ny() const noexcept { return box_.ny(); }

  Field2D<Cell>& field() noexcept { return field_; }
  const Field2D<Cell>& field() const noexcept { return field_; }

 private:
  Box2D box_;
  Field2D<Cell> field_;
};

}  // namespace pde_core
