#pragma once

#include <pde_core/amr/box2d.hpp>
#include <pde_core/core/types.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <vector>

namespace pde_core {

// Berger-Rigoutsos (1991) point clustering : given a 2D boolean mark
// grid (marks[j*nx + i] is true if cell (i, j) is tagged for refinement),
// return a list of rectangular boxes that cover all tagged cells with
// a minimum "efficiency" (ratio of tagged cells to total cells in each
// box).
//
// Reference : Berger & Rigoutsos (1991), "An algorithm for point
// clustering and grid generation", IEEE Trans. Systems, Man, and
// Cybernetics 21(5):1278-1286.
//
// Algorithm sketch :
//   1. Take the bounding box of the tagged cells in the input region.
//   2. If efficiency >= threshold, accept the box as-is.
//   3. Else, look for a HOLE (a row/column with zero tagged cells) and
//      split there. Holes give the cleanest cuts.
//   4. If no hole, look for an INFLECTION POINT in the row/column count
//      signatures (where the second derivative changes sign with
//      maximum magnitude) and split there.
//   5. Recurse on each half. Stop subdividing when the box becomes too
//      small (< 2 * min_box_size in both dimensions).
//
// Output boxes have level = 0 (caller assigns the level later).

struct ClusteringParams2D {
  Real efficiency_threshold = Real{0.7};
  std::ptrdiff_t min_box_size = 2;
};

namespace detail {

// Find the best hole (zero entry in S) to split at. Returns -1 if no
// valid hole. "Valid" means the split [0, k) | [k, n) both have at
// least `min_size` entries.
inline std::ptrdiff_t cluster_find_hole(const std::vector<std::ptrdiff_t>& S,
                                         std::ptrdiff_t min_size) {
  const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(S.size());
  const std::ptrdiff_t mid = n / 2;
  std::ptrdiff_t best = -1;
  std::ptrdiff_t best_dist = 0;
  for (std::ptrdiff_t k = min_size; k <= n - min_size; ++k) {
    if (S[k] == 0) {
      const std::ptrdiff_t dist = std::abs(k - mid);
      if (best < 0 || dist < best_dist) {
        best = k;
        best_dist = dist;
      }
    }
  }
  return best;
}

// Find the best inflection point (in the second derivative of S). Splits
// between S[k] and S[k+1] where sign(D[k-1]) != sign(D[k]). Returns the
// split position (in [min_size, n - min_size]) where the magnitude of
// the change in D is largest, or -1.
inline std::ptrdiff_t cluster_find_inflection(
    const std::vector<std::ptrdiff_t>& S, std::ptrdiff_t min_size) {
  const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(S.size());
  if (n < 4) return -1;
  // D[k] = S[k+1] - 2*S[k] + S[k-1] for k in [1, n-1). Stored
  // 0-indexed : D[k - 1] is D evaluated at S-index k.
  std::vector<std::ptrdiff_t> D(static_cast<std::size_t>(n - 2));
  for (std::ptrdiff_t k = 1; k < n - 1; ++k) {
    D[static_cast<std::size_t>(k - 1)] = S[k + 1] - 2 * S[k] + S[k - 1];
  }
  std::ptrdiff_t best = -1;
  std::ptrdiff_t best_mag = 0;
  // Sign change between D[k - 1] (at S-index k) and D[k] (at S-index k+1).
  // Split at S-index k+1.
  for (std::ptrdiff_t k = 1; k < static_cast<std::ptrdiff_t>(D.size()); ++k) {
    const auto a = D[static_cast<std::size_t>(k - 1)];
    const auto b = D[static_cast<std::size_t>(k)];
    const bool sign_change = (a > 0 && b < 0) || (a < 0 && b > 0);
    const std::ptrdiff_t split = k + 1;  // split between S[k] and S[k+1]
    if (sign_change && split >= min_size && split <= n - min_size) {
      const auto mag = std::abs(b - a);
      if (mag > best_mag) {
        best_mag = mag;
        best = split;
      }
    }
  }
  return best;
}

inline void cluster_recurse(const std::vector<bool>& marks,
                             std::ptrdiff_t nx, std::ptrdiff_t /*ny*/,
                             std::ptrdiff_t i_lo, std::ptrdiff_t i_hi,
                             std::ptrdiff_t j_lo, std::ptrdiff_t j_hi,
                             const ClusteringParams2D& params,
                             std::vector<Box2D>& out) {
  // Tighten the box to the actual extent of marked cells in the
  // sub-region (drops the empty rows/cols on every recursion).
  std::ptrdiff_t i_min = i_hi, i_max = i_lo - 1;
  std::ptrdiff_t j_min = j_hi, j_max = j_lo - 1;
  std::ptrdiff_t count = 0;
  for (std::ptrdiff_t j = j_lo; j < j_hi; ++j) {
    for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) {
      if (marks[static_cast<std::size_t>(j * nx + i)]) {
        ++count;
        if (i < i_min) i_min = i;
        if (i > i_max) i_max = i;
        if (j < j_min) j_min = j;
        if (j > j_max) j_max = j;
      }
    }
  }
  if (count == 0) return;
  const std::ptrdiff_t bx_lo = i_min, bx_hi = i_max + 1;
  const std::ptrdiff_t by_lo = j_min, by_hi = j_max + 1;
  const std::ptrdiff_t bxw = bx_hi - bx_lo;
  const std::ptrdiff_t byw = by_hi - by_lo;
  const Real eff = static_cast<Real>(count) /
                    static_cast<Real>(bxw * byw);
  const bool too_small_to_split = (bxw < 2 * params.min_box_size) &&
                                   (byw < 2 * params.min_box_size);
  if (eff >= params.efficiency_threshold || too_small_to_split) {
    out.push_back(Box2D{bx_lo, bx_hi, by_lo, by_hi, 0});
    return;
  }

  // Compute signatures Sx (along i), Sy (along j).
  std::vector<std::ptrdiff_t> Sx(static_cast<std::size_t>(bxw), 0);
  std::vector<std::ptrdiff_t> Sy(static_cast<std::size_t>(byw), 0);
  for (std::ptrdiff_t j = by_lo; j < by_hi; ++j) {
    for (std::ptrdiff_t i = bx_lo; i < bx_hi; ++i) {
      if (marks[static_cast<std::size_t>(j * nx + i)]) {
        Sx[static_cast<std::size_t>(i - bx_lo)] += 1;
        Sy[static_cast<std::size_t>(j - by_lo)] += 1;
      }
    }
  }

  // Look for a hole, then for an inflection. Holes give priority because
  // they're a clean cut with no efficiency loss.
  std::ptrdiff_t holex = -1, holey = -1;
  if (bxw >= 2 * params.min_box_size)
    holex = cluster_find_hole(Sx, params.min_box_size);
  if (byw >= 2 * params.min_box_size)
    holey = cluster_find_hole(Sy, params.min_box_size);

  int split_axis = -1;            // 0 = x, 1 = y
  std::ptrdiff_t split_pos = -1;
  if (holex >= 0 && holey >= 0) {
    split_axis = (bxw >= byw) ? 0 : 1;
    split_pos = (split_axis == 0) ? holex : holey;
  } else if (holex >= 0) {
    split_axis = 0;
    split_pos = holex;
  } else if (holey >= 0) {
    split_axis = 1;
    split_pos = holey;
  } else {
    std::ptrdiff_t infx = -1, infy = -1;
    if (bxw >= 2 * params.min_box_size)
      infx = cluster_find_inflection(Sx, params.min_box_size);
    if (byw >= 2 * params.min_box_size)
      infy = cluster_find_inflection(Sy, params.min_box_size);
    if (infx >= 0 && infy >= 0) {
      split_axis = (bxw >= byw) ? 0 : 1;
      split_pos = (split_axis == 0) ? infx : infy;
    } else if (infx >= 0) {
      split_axis = 0;
      split_pos = infx;
    } else if (infy >= 0) {
      split_axis = 1;
      split_pos = infy;
    }
  }

  if (split_axis < 0) {
    out.push_back(Box2D{bx_lo, bx_hi, by_lo, by_hi, 0});
    return;
  }

  if (split_axis == 0) {
    const std::ptrdiff_t i_split = bx_lo + split_pos;
    cluster_recurse(marks, nx, 0, bx_lo, i_split, by_lo, by_hi, params, out);
    cluster_recurse(marks, nx, 0, i_split, bx_hi, by_lo, by_hi, params, out);
  } else {
    const std::ptrdiff_t j_split = by_lo + split_pos;
    cluster_recurse(marks, nx, 0, bx_lo, bx_hi, by_lo, j_split, params, out);
    cluster_recurse(marks, nx, 0, bx_lo, bx_hi, j_split, by_hi, params, out);
  }
}

}  // namespace detail

inline std::vector<Box2D> cluster_berger_rigoutsos(
    const std::vector<bool>& marks,
    std::ptrdiff_t nx, std::ptrdiff_t ny,
    const ClusteringParams2D& params = ClusteringParams2D{}) {
  std::vector<Box2D> out;
  detail::cluster_recurse(marks, nx, ny, 0, nx, 0, ny, params, out);
  return out;
}

}  // namespace pde_core
