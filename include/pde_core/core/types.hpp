#pragma once

#include <cstddef>

namespace pde_core {

// Floating-point precision for the whole core. Consumers can override by
// defining PDE_CORE_USE_FLOAT before including any pde_core header.
#ifdef PDE_CORE_USE_FLOAT
using Real = float;
#else
using Real = double;
#endif

// Signed integer index type. ptrdiff_t to allow negative ghost-cell
// indexing (i.e. U(-1, j) accesses the left ghost layer).
using Index = std::ptrdiff_t;

}  // namespace pde_core
