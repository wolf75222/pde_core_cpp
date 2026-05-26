#pragma once

#include <pde_core/core/cell_traits.hpp>
#include <pde_core/core/types.hpp>

#include <cstddef>
#include <utility>
#include <vector>

namespace pde_core {

// 2D cell-centred field, row-major (x is the fast-varying index), with
// `ghost` ghost cells on all four sides. Templated on the cell type :
//   - scalar problems : Field2D<Real>
//   - Euler 2D        : Field2D<ConservedState2D> via consumer alias
//
// Interior indices : (i, j) with i in [0, nx), j in [0, ny). Negative or
// out-of-range indices access ghost cells.
template <class Cell>
class Field2D {
 public:
  Field2D(Index nx, Index ny, Index ghost = 2)
      : nx_(nx), ny_(ny), ghost_(ghost),
        stride_(nx + 2 * ghost),
        data_(static_cast<std::size_t>(stride_ * (ny + 2 * ghost)),
              cell_zero<Cell>()) {}

  Index nx() const noexcept { return nx_; }
  Index ny() const noexcept { return ny_; }
  Index ghost() const noexcept { return ghost_; }

  Cell& operator()(std::ptrdiff_t i, std::ptrdiff_t j) noexcept {
    return data_[linear(i, j)];
  }
  const Cell& operator()(std::ptrdiff_t i, std::ptrdiff_t j) const noexcept {
    return data_[linear(i, j)];
  }

  Cell* data() noexcept { return data_.data(); }
  const Cell* data() const noexcept { return data_.data(); }

  friend void swap(Field2D& a, Field2D& b) noexcept {
    using std::swap;
    swap(a.nx_, b.nx_);
    swap(a.ny_, b.ny_);
    swap(a.ghost_, b.ghost_);
    swap(a.stride_, b.stride_);
    swap(a.data_, b.data_);
  }

 private:
  std::size_t linear(std::ptrdiff_t i, std::ptrdiff_t j) const noexcept {
    const auto g = static_cast<std::ptrdiff_t>(ghost_);
    const auto s = static_cast<std::ptrdiff_t>(stride_);
    return static_cast<std::size_t>((j + g) * s + (i + g));
  }

  Index nx_, ny_, ghost_, stride_;
  std::vector<Cell> data_;
};

}  // namespace pde_core
