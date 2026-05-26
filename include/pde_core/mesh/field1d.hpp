#pragma once

#include <pde_core/core/cell_traits.hpp>
#include <pde_core/core/types.hpp>

#include <cstddef>
#include <utility>
#include <vector>

namespace pde_core {

// 1D cell-centred field with `ghost` ghost cells on both sides. Templated on
// the cell type :
//   - scalar problems : Field1D<Real>
//   - systems         : Field1D<Eigen::Matrix<Real, N, 1>> via consumer alias
//                       (Euler 1D uses N = 3).
//
// AoS layout : data_[i + ghost] holds cell i. Signed-index operator() makes
// stencil loops natural (i in [-ghost, nx + ghost)).
template <class Cell>
class Field1D {
 public:
  explicit Field1D(Index nx, Index ghost = 2)
      : nx_(nx), ghost_(ghost),
        data_(static_cast<std::size_t>(nx + 2 * ghost), cell_zero<Cell>()) {}

  Index nx() const noexcept { return nx_; }
  Index ghost() const noexcept { return ghost_; }
  Index total_size() const noexcept { return nx_ + 2 * ghost_; }

  Cell& operator()(std::ptrdiff_t i) noexcept {
    return data_[static_cast<std::size_t>(
        static_cast<std::ptrdiff_t>(ghost_) + i)];
  }
  const Cell& operator()(std::ptrdiff_t i) const noexcept {
    return data_[static_cast<std::size_t>(
        static_cast<std::ptrdiff_t>(ghost_) + i)];
  }

  Cell* data() noexcept { return data_.data(); }
  const Cell* data() const noexcept { return data_.data(); }

  friend void swap(Field1D& a, Field1D& b) noexcept {
    using std::swap;
    swap(a.nx_, b.nx_);
    swap(a.ghost_, b.ghost_);
    swap(a.data_, b.data_);
  }

 private:
  Index nx_;
  Index ghost_;
  std::vector<Cell> data_;
};

}  // namespace pde_core
