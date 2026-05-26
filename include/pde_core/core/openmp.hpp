#pragma once

// OpenMP-aware pragma helpers. Enabled when the consumer build defines
// PDE_CORE_HAS_OPENMP (which the pde_core CMake INTERFACE library exposes
// when PDE_CORE_USE_OPENMP=ON, but consumers can also set it themselves
// alongside their own OpenMP linkage).
//
// Use site:
//   PDE_OMP_PARALLEL_FOR  -- wraps `#pragma omp parallel for` (or nothing)
//   PDE_OMP_CRITICAL      -- wraps `#pragma omp critical` (or nothing)
//
// Both expand to nothing when OpenMP is not available, so the code stays
// portable.
//
// Example :
//   PDE_OMP_PARALLEL_FOR
//   for (Index j = 0; j < ny; ++j) { ... }
//
//   PDE_OMP_CRITICAL
//   { shared_accumulator += local; }

#ifdef PDE_CORE_HAS_OPENMP
  #include <omp.h>
  #define PDE_OMP_PARALLEL_FOR _Pragma("omp parallel for")
  #define PDE_OMP_CRITICAL     _Pragma("omp critical")
#else
  #define PDE_OMP_PARALLEL_FOR
  #define PDE_OMP_CRITICAL
#endif

namespace pde_core {

// Convenience accessors that work whether OpenMP is enabled or not.
inline int omp_num_threads() noexcept {
#ifdef PDE_CORE_HAS_OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

inline int omp_thread_id() noexcept {
#ifdef PDE_CORE_HAS_OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

}  // namespace pde_core
