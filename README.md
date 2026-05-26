<div align="center">

# PDE Core CPP

**Infrastructure C++20 partagée pour les solveurs PDE : grilles, champs templates, primitives AMR, clustering Berger-Rigoutsos.**

![C++20](https://img.shields.io/badge/C%2B%2B-20-blue?logo=cplusplus)
![Header-only](https://img.shields.io/badge/header--only-yes-success)
![Build](https://img.shields.io/badge/build-CMake%203.20%2B-064F8C?logo=cmake)
![License](https://img.shields.io/badge/license-BSD--3-green)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey)

</div>

---

`pde_core_cpp` est la bibliothèque header-only qui héberge les briques
infrastructure communes à [`advection_cpp`](https://github.com/wolf75222/advection_cpp)
(advection scalaire + Burgers) et [`euler_cpp`](https://github.com/wolf75222/euler_cpp)
(Euler 2D compressible et extensions Navier-Stokes / plasma). Elle est
volontairement type-agnostique : tout ce qui dépendrait du système
d'équations conservatives concret reste dans le solveur consommateur.

L'objectif : éviter qu'un correctif sur Berger-Rigoutsos ou sur le
layout `Field2D` doive être propagé manuellement entre deux dépôts.
Les solveurs consomment `pde_core` via `FetchContent`, ré-exportent
les symboles dans leur propre namespace (`advection::Field2D`,
`euler::Field2D`) et utilisent leurs alias historiques sans rien
casser côté API.

## Modules

### Coeur et primitives mesh

| Module | En-tête | Rôle |
|---|---|---|
| **types** | [`pde_core/core/types.hpp`](include/pde_core/core/types.hpp) | `Real` (double, surchargeable via `PDE_CORE_USE_FLOAT`), `Index` (`std::ptrdiff_t` signé) |
| **openmp** | [`pde_core/core/openmp.hpp`](include/pde_core/core/openmp.hpp) | Macros `PDE_OMP_PARALLEL_FOR` / `PDE_OMP_CRITICAL`, neutralisées hors OpenMP |
| **cell_traits** | [`pde_core/core/cell_traits.hpp`](include/pde_core/core/cell_traits.hpp) | `CellTraits<Cell>::zero()` qui détecte `Cell::Zero()` par SFINAE, accesseur composante `cell_at(c, k)`, et `kahan_add<Cell>` componentwise |
| **Domain1D / Domain2D** | [`pde_core/mesh/domain*.hpp`](include/pde_core/mesh/) | Grilles uniformes cell-centered, accesseurs `dx()`, `x_cell(i)` |
| **Field1D / Field2D** | [`pde_core/mesh/field*.hpp`](include/pde_core/mesh/) | Champs cell-centered AoS avec ghost cells, templated `<Cell>`, accès `U(i, j)` signé |

### Conditions aux limites communes

| Module | En-tête | Rôle |
|---|---|---|
| **PeriodicBC1D / 2D** | [`pde_core/bc/periodic*.hpp`](include/pde_core/bc/) | BC périodique avec corner pass pour les 4 coins en 2D |
| **OutflowBC1D / 2D** | [`pde_core/bc/outflow*.hpp`](include/pde_core/bc/) | Extrapolation zero-gradient transmissive |

### AMR block-structuré

| Module | En-tête | Rôle |
|---|---|---|
| **Box1D / Box2D** | [`pde_core/amr/box*.hpp`](include/pde_core/amr/) | Rectangles d'index dans l'espace de raffinement, ratio fixé à 2 |
| **MeshBlock1D / 2D** | [`pde_core/amr/mesh_block*.hpp`](include/pde_core/amr/) | Box + Field, unité de travail à la Athena++, templated `<Cell>` |
| **MeshHierarchy1D / 2D** | [`pde_core/amr/mesh_hierarchy*.hpp`](include/pde_core/amr/) | Hiérarchie multi-niveau (jusqu'à 8) + multi-patch (deque pour stabilité des références), API héritée à 2 niveaux conservée pour la rétrocompat |
| **FluxRegister1D / 2D** | [`pde_core/amr/flux_register*.hpp`](include/pde_core/amr/) | Bookkeeping du refluxing Berger-Colella, somme Kahan-compensée pour stabilité bit-à-bit cross-plateforme, variantes weighted SSPRK3, per-thread accumulators avec merge |
| **clustering2d** | [`pde_core/amr/clustering2d.hpp`](include/pde_core/amr/clustering2d.hpp) | Algorithme de Berger-Rigoutsos (1991) qui couvre les cellules marquées par des rectangles |
| **ghost_fill2d** | [`pde_core/amr/ghost_fill2d.hpp`](include/pde_core/amr/ghost_fill2d.hpp) | Remplissage des ghost cells multi-patch : parent avec interpolation temporelle + override sibling pour les voisins de même niveau |
| **regrid1d / 2d** | [`pde_core/amr/regrid*.hpp`](include/pde_core/amr/) | Régénération du patch fin autour des cellules marquées, variantes single-block, multi-niveau (proper nesting), et multi-patch (via clustering) |

Le détail de chaque algorithme, les conventions d'indexation et les
références bibliographiques sont dans
[`docs/ALGORITHMS.md`](docs/ALGORITHMS.md) et
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Écosystème

`pde_core_cpp` est l'une de quatre bibliothèques C++20 qui forment un écosystème de solveurs PDE :

| Repo | Rôle | Dépend de |
|---|---|---|
| [`poisson_cpp`](https://github.com/wolf75222/poisson_cpp) | Solveurs Poisson (Thomas, SOR, CG, DST spectral, AMR + multigrille) | indépendant |
| **`pde_core_cpp`** (ce dépôt) | Infrastructure C++ partagée entre les solveurs hyperboliques : mesh, fields templés `<Cell>`, AMR primitives, clustering Berger-Rigoutsos, BC périodique/outflow | indépendant |
| [`advection_cpp`](https://github.com/wolf75222/advection_cpp) | Advection scalaire + Burgers + advection rotative (MUSCL, WENO5) + Chorin NS incompressible (opt-in) | `pde_core_cpp`, `poisson_cpp` (opt-in) |
| [`euler_cpp`](https://github.com/wolf75222/euler_cpp) | Euler 2D compressible + viscous NS (no-slip walls) + sources plasma (Lorentz, Hall) + Euler-Poisson AMR self-gravity | `pde_core_cpp`, `poisson_cpp` (opt-in) |

`pde_core_cpp` ne dépend de rien et est tirée par les deux solveurs hyperboliques via `FetchContent`. La séparation a été motivée par un audit qui a mesuré ~2000 lignes d'infrastructure dupliquées entre `advection_cpp` et `euler_cpp` : ce dépôt héberge la version canonique des briques type-agnostiques.

## Utilisation depuis un solveur consommateur

```cmake
include(FetchContent)
FetchContent_Declare(
  pde_core
  GIT_REPOSITORY https://github.com/wolf75222/pde_core_cpp.git
  GIT_TAG main)
FetchContent_MakeAvailable(pde_core)

target_link_libraries(my_solver PUBLIC pde_core::pde_core)
```

Pendant le développement, pour itérer sans cycle commit/push/fetch,
passer le chemin local en cache :

```bash
cmake -B build -DPDE_CORE_SOURCE_DIR=$HOME/Documents/Stage_Romain/pde_core_cpp
```

Côté code, le templating sur `Cell` couvre uniformément les champs
scalaires et systèmes :

```cpp
#include <pde_core/mesh/field2d.hpp>
#include <pde_core/amr/clustering2d.hpp>

// Champ scalaire (advection, Burgers)
pde_core::Field2D<double> u(64, 64, /*ghost=*/2);

// Champ système (Euler 2D : 4-vecteur)
using ConservedState2D = Eigen::Matrix<double, 4, 1>;
pde_core::Field2D<ConservedState2D> U(64, 64, /*ghost=*/2);
// Les cellules sont automatiquement initialisées via
// ConservedState2D::Zero() détecté par CellTraits.

// Clustering pour le regrid AMR
std::vector<bool> marks(nx * ny, false);
// ... marquer les cellules à raffiner ...
auto patches = pde_core::cluster_berger_rigoutsos(marks, nx, ny);
```

Le détail du templating et de la dispatch `CellTraits` est dans
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#templating-sur-cell).

## Build et tests

```bash
git clone https://github.com/wolf75222/pde_core_cpp.git
cd pde_core_cpp
cmake -B build -DPDE_CORE_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Options CMake :

| Option | Défaut | Rôle |
|---|---|---|
| `PDE_CORE_BUILD_TESTS` | `OFF` | Compile la suite Catch2 |
| `PDE_CORE_USE_OPENMP` | `OFF` | Active OpenMP transitivement (définit `PDE_CORE_HAS_OPENMP`) |

Les tests vérifient :

- L'initialisation à zéro de `Field<Cell>` pour `Real` et pour un type
  Eigen-like (via une fausse cellule avec méthode statique `Zero()`,
  pour confirmer que la SFINAE de `CellTraits` capte les deux familles).
- L'accès ghost cells par index négatif sur `Field1D` et `Field2D`.
- Les coordonnées cell-centered de `Domain1D` et `Domain2D`.
- La cohérence `MeshBlock` ↔ `Field` (taille, ghost, indexation).
- Berger-Rigoutsos sur trois cas : grille vide, un bloc de cellules
  marquées, deux blocs disjoints (vérifie que l'algo coupe sur le
  trou). Les boîtes retournées couvrent exactement la surface attendue.

## Intégration CI

GitHub Actions construit et exécute la suite sur `ubuntu-latest` et
`macos-latest` à chaque push. Voir [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## License

BSD-3-Clause. Voir [LICENSE](LICENSE).
