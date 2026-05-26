# pde_core_cpp

Shared header-only C++20 core for the `advection_cpp` and `euler_cpp`
solvers. Holds the type-agnostic infrastructure that was being copied
between the two repos : mesh, fields templated on cell type, AMR
index-space primitives, Berger-Rigoutsos clustering.

## Status

Pass 1 (this repo) :

- `pde_core/core/{types, openmp, cell_traits}.hpp`
- `pde_core/mesh/{domain, field}{1d,2d}.hpp` (field templated on `Cell`)
- `pde_core/amr/{box, mesh_block}{1d,2d}.hpp`
- `pde_core/amr/clustering2d.hpp` (Berger-Rigoutsos 1991)

Deferred to a later pass (kept per-repo because they depend on
solver-specific scheme + payload types) :

- `mesh_hierarchy{1d,2d}`, `flux_register{1d,2d}`, `regrid{1d,2d}`,
  `ghost_fill2d`, `amr_integrator*`
- time integrators (`explicit_euler`, `ssp_rk2`, `ssp_rk3`, Strang split)
- boundary conditions (`periodic`, `outflow`, `dirichlet`)

## Use from a consumer repo

```cmake
include(FetchContent)
FetchContent_Declare(
  pde_core_cpp
  GIT_REPOSITORY https://github.com/wolf75222/pde_core_cpp.git
  GIT_TAG main)
FetchContent_MakeAvailable(pde_core_cpp)

target_link_libraries(my_solver PUBLIC pde_core::pde_core)
```

Then in code :

```cpp
#include <pde_core/mesh/field2d.hpp>
#include <pde_core/amr/clustering2d.hpp>

// Scalar field
pde_core::Field2D<double> u(64, 64, /*ghost=*/2);

// Or system field (Euler 2D : 4-vector)
using ConservedState2D = Eigen::Matrix<double, 4, 1>;
pde_core::Field2D<ConservedState2D> U(64, 64, /*ghost=*/2);
```

The `Field*D<Cell>` initialisation uses `CellTraits<Cell>::zero()`, which
detects `Cell::Zero()` automatically for Eigen matrix / array types and
falls back to value-initialisation otherwise.

## Build the library tests

```bash
cmake -S . -B build -DPDE_CORE_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build
```

## License

Same as the consumer projects.
