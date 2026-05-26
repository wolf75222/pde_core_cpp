# Algorithmes

Description et pseudocode des structures et algorithmes que `pde_core_cpp`
fournit aux solveurs consommateurs. Chaque section est rattachée au
fichier d'en-tête correspondant.

## 1. Grille uniforme cell-centered (`Domain1D`, `Domain2D`)

Le domaine `[x_min, x_max]` est partitionné en `nx` cellules de largeur
`dx = (x_max - x_min) / nx`. Les valeurs sont stockées au centre :

```
x_cell(i) = x_min + (i + 1/2) * dx,    i = 0, 1, ..., nx - 1
```

Le `+ 1/2` est l'élément qui rend la discrétisation finie-volume
naturellement conservative : un flux numérique entre `(i, i+1)` est
défini à la face `x_face(i+1/2) = x_min + (i+1) * dx`, et la dérivée
discrète `(F_{i+1/2} - F_{i-1/2}) / dx` approxime exactement la
moyenne cellulaire de `∂_x F`.

`Domain2D` est l'extension tensor-product avec `dy`, `y_cell(j)`. Aucune
allocation : ce sont des `struct` triviaux passés par valeur. Tous les
accesseurs sont `constexpr` pour permettre des calculs de temps de
compilation (boucles déroulées, tailles de buffer).

## 2. Champ cell-centered avec ghost cells (`Field1D<Cell>`, `Field2D<Cell>`)

### Layout mémoire

`Field2D` stocke un tableau `std::vector<Cell>` de taille
`(nx + 2g) * (ny + 2g)` en **AoS row-major**, c'est-à-dire `x` est
l'index rapide :

```
stride = nx + 2 * ghost
data[(j + g) * stride + (i + g)]  ←→  U(i, j)
```

Les indices `(i, j)` exposés à l'utilisateur sont **signés**
(`std::ptrdiff_t`). L'intervalle physique des cellules intérieures est
`i ∈ [0, nx)`, `j ∈ [0, ny)`. Les valeurs `i = -1, -2, ..., -g` (resp.
`i = nx, nx + 1, ..., nx + g - 1`) accèdent les `g` couches de ghost
cells à gauche (resp. à droite). Cette convention rend les stencils
naturels :

```cpp
// upwind 1D, schéma F(U_i) = a * U_i, vitesse a > 0
for (Index i = 0; i < nx; ++i)
  U_new(i) = U(i) - (dt / dx) * a * (U(i) - U(i - 1));
```

Pas de wrap-around, pas de `std::max(0, i - 1)`, pas de branchement.
C'est aussi ce qui justifie le choix de `Index = std::ptrdiff_t` (signé)
au lieu de `std::size_t`.

### Templating sur `Cell`

`Field1D` et `Field2D` sont templés sur le type de la cellule :

- `Field2D<Real>` pour un champ scalaire (advection, Burgers, diffusion).
- `Field2D<Eigen::Matrix<Real, N, 1>>` pour un système conservatif à
  N composantes (Euler 1D : N = 3 ; Euler 2D : N = 4 ; MHD ideale : N = 8).

L'initialisation à zéro se fait via le trait `CellTraits<Cell>::zero()`,
qui détecte par SFINAE la présence d'une méthode statique `Cell::Zero()` :

```cpp
template <class Cell, class = void>
struct CellTraits {
  static constexpr Cell zero() noexcept(noexcept(Cell{})) { return Cell{}; }
};

// Spécialisation auto pour Eigen::Matrix et tout type avec Cell::Zero()
template <class Cell>
struct CellTraits<Cell, std::void_t<decltype(Cell::Zero())>> {
  static auto zero() noexcept(noexcept(Cell::Zero())) { return Cell::Zero(); }
};
```

Le consommateur n'a rien à déclarer : `Field2D<Real>` retombe sur
`Real{}` (= `0.0`), `Field2D<Eigen::Matrix<Real, 4, 1>>` détecte
automatiquement `Matrix::Zero()` et initialise toutes les cellules à
`(0, 0, 0, 0)`. Aucun risque d'oubli d'init.

## 3. Box, MeshBlock (`Box1D/2D`, `MeshBlock1D/2D<Cell>`)

### `Box` : géométrie d'index

`Box2D` décrit un rectangle dans l'espace d'index d'un niveau de
raffinement. C'est de la géométrie pure, sans données :

```cpp
struct Box2D {
  std::ptrdiff_t i_lo, i_hi, j_lo, j_hi;   // [i_lo, i_hi) x [j_lo, j_hi)
  int level;                               // niveau AMR
};
```

Convention : un Box au niveau `l` utilise des indices sur la grille
de pas `dx_l = dx_0 / r^l` avec `r = 2` (ratio de raffinement fixé).
Pour mapper un indice fin sur sa cellule parente : `i_parent = i_fine / r`
(division entière, signed pour gérer les `i < 0`).

### `MeshBlock` : Box + Field

Un `MeshBlock` est l'unité de travail à la Athena++ / BoxLib :

```cpp
template <class Cell>
class MeshBlock2D {
  Box2D box_;
  Field2D<Cell> field_;
public:
  MeshBlock2D(Box2D box, Index ghost = 2)
    : box_(box), field_(box.nx(), box.ny(), ghost) {}
  // ...
};
```

Toujours alloué avec les `ghost` couches sur les 4 côtés. Le ghost
count est uniforme sur tous les niveaux ; les variantes "ghost minces
en coarse" sont gérées par l'intégrateur AMR du consommateur, pas par
`pde_core`.

Le `pde_core` ne stocke pas la hiérarchie de blocs : `MeshHierarchy`
reste dans chaque solveur, parce que la dispatch flux register +
refluxing + step subcyclé dépend fortement du type d'état (`Real` vs
`ConservedState2D`) et du choix de schéma de flux. La hiérarchie
consomme et compose des `MeshBlock<Cell>` mais n'est pas elle-même
type-agnostique.

## 4. Clustering Berger-Rigoutsos (`clustering2d`)

### Problème

Étant donné une grille 2D booléenne `marks[j * nx + i]` qui marque les
cellules à raffiner (sortie d'un critère type `|∇ρ| > seuil`), produire
une liste de boîtes rectangulaires qui :

1. couvrent toutes les cellules marquées,
2. ont une efficacité (= ratio cellules marquées / cellules totales)
   au-dessus d'un seuil (par défaut 0.7),
3. ne sont pas plus petites qu'une taille minimale dans aucune
   dimension (par défaut 2 cellules).

C'est le problème central du regrid dans Berger-Colella AMR (1989) :
on veut peu de boîtes mais bien remplies, pour ne pas gaspiller des
cellules raffinées sur des zones inactives.

### Algorithme (Berger & Rigoutsos, 1991)

L'algorithme est récursif sur la sous-région courante :

```
function cluster(marks, region):
    box = bounding_box(marks ∩ region)
    if efficiency(box, marks) ≥ threshold:
        emit box; return
    if box.width < 2 * min_size AND box.height < 2 * min_size:
        emit box; return                    # trop petit pour couper

    # Calcul des signatures (projections orthogonales)
    Sx[i] = nb de cellules marquées dans la colonne i
    Sy[j] = nb de cellules marquées dans la ligne j

    # Stratégie 1 : couper sur un TROU (Sx[k] = 0 ou Sy[k] = 0)
    holex = find_hole(Sx, min_size)         # split le plus proche du centre
    holey = find_hole(Sy, min_size)
    if hole found:
        split sur l'axe le plus long, à holex ou holey
        recurse sur les deux moitiés
        return

    # Stratégie 2 : couper à un INFLEXION dans la dérivée seconde
    # D[k] = Sx[k+1] - 2*Sx[k] + Sx[k-1]
    # split là où sign(D[k-1]) ≠ sign(D[k]) avec |D[k] - D[k-1]| maximal
    infx = find_inflection(Sx, min_size)
    infy = find_inflection(Sy, min_size)
    if inflection found:
        split sur l'axe le plus long, à infx ou infy
        recurse sur les deux moitiés
        return

    # Fallback : aucun split valable, on accepte la box telle quelle
    emit box
```

Les trous sont privilégiés parce qu'ils donnent une coupe à efficacité
préservée (le `0` dans la signature signifie qu'on peut séparer la
région sans inclure de cellule vide supplémentaire). Quand il n'y en
a pas, l'inflexion détecte le "point de pliure" où deux clusters
distincts se rejoignent par leurs queues.

### Complexité

Pour `N = nx * ny` cellules et `B` boîtes en sortie :

- Calcul de la bounding box : `O(N)` par récursion.
- Calcul des signatures : `O(N)` par récursion.
- Recherche du trou / inflexion : `O(max(nx, ny))`.
- Récursion : `O(log B)` niveaux en moyenne.

Coût total empirique : `O(N log B)`, ce qui est bien plus rapide que
le coût des `B` solveurs AMR exécutés sur les boîtes — donc le
clustering n'est jamais un bottleneck.

### Référence

Berger M.J., Rigoutsos I. (1991), *"An algorithm for point clustering
and grid generation"*, IEEE Transactions on Systems, Man, and
Cybernetics 21(5), 1278-1286.

### Test

`tests/test_clustering.cpp` :

- Grille vide → 0 box.
- Un seul rectangle marqué (5..11, 4..9) → 1 box, qui couvre exactement
  (5..11, 4..9).
- Deux rectangles disjoints séparés par 10 colonnes vides → 2 box,
  surface totale = 2 × 4 × 4 = 32 cellules (vérifié exactement).

## 5. MeshHierarchy (`mesh_hierarchy{1d,2d}`)

`MeshHierarchy2D<Cell>` est le conteneur multi-niveau / multi-patch de
`MeshBlock<Cell>`. Il accepte jusqu'à `max_level = 8` niveaux nichés,
chacun avec un nombre arbitraire de patches. Conventions :

- Le niveau 0 est la grille coarse fixée à la construction (un seul bloc
  qui couvre tout le domaine).
- Le niveau `L` est composé de patches placés via
  `set_patches_at_level(L, covs)` ou `add_patch_at_level(L, cov)`.
- Une coverage `cov.i_lo, i_hi, j_lo, j_hi` est exprimée dans l'espace
  de cellules du parent (niveau `L - 1`). Les coordonnées Box d'un
  patch à niveau `L` sont en coordonnées de niveau `L` (donc multipliées
  par `r = 2` à chaque descente).
- Le stockage interne `patches_[L-1]` est un `std::deque<MeshBlock2D<Cell>>`
  plutôt qu'un `std::vector`, ce qui garantit la stabilité des
  références à travers les `push_back` ultérieurs (un `add_patch` ne
  doit pas invalider une référence retenue par l'intégrateur).
- L'API héritée à 2 niveaux (`has_fine()`, `fine()`, `fine_lo_*_coarse()`)
  est conservée comme miroir vers le premier patch de niveau 1, pour
  que les tests historiques restent bit-équivalents.

`refine_at_level(L, ...)` réinitialise tous les niveaux au-dessus de L
(ils deviennent orphelins). C'est le point d'entrée standard du regrid.

## 6. FluxRegister (`flux_register{1d,2d}`) avec somme Kahan

### Le problème de la dérive de masse

Dans Berger-Colella 1989, le coarse step utilise la flux face coarse
$F^c$ aux interfaces coarse/fine. La fine patch fait `r = 2` sub-steps
et accumule au passage les `r * r` fluxes fines aux faces du même
sommet. À la fin du coarse step, on remplace $F^c$ par la moyenne
fine $\overline{F^f}$ et on propage la correction dans la cellule
coarse adjacente. Le `+=` sur l'accumulateur fine doit être stable
bit-à-bit pour que le résultat ne dépende pas de l'ordre dans lequel
les threads (en OpenMP) écrivent leurs contributions, ni de
l'associativité que le compilo applique sous `-ffast-math`.

La somme naïve `sum += x` perd $\sim \epsilon \cdot |sum|$ par addition.
Sur des milliers de coarse steps et `r * r = 4` contributions par slot,
ça accumule en `~ N_steps * eps * |F|` de bias systématique, observable
comme `~0.1%/step` de dérive de masse (mesure faite sur ubuntu-latest
gcc avec SSE2 vs macOS clang avec NEON).

### L'algorithme de Kahan (1965)

Kahan introduit un compensation term `c` qui mémorise la partie
basse perdue à chaque add :

```
y = x − c              // compense la partie perdue à l'add précédent
t = sum + y            // add lossy
c = (t − sum) − y      // bits perdus à cet add (les y bas qui n'ont pas
                       //  trouvé leur place dans la mantisse de t)
sum = t
```

L'erreur de troncature passe de `O(eps)` à `O(eps^2)`, et surtout le
résultat devient **invariant à l'ordre** des additions à parité de
contributions. Sur la dérive de masse, le bias passe de `~1e-3/step`
à `~1e-15/step` = précision machine.

### Spécificité C++ : `volatile` pour défaire la réassociation

Un compilo qui optimise `(t - sum) - y` peut le réécrire en
`t - (sum + y) = t - t = 0` par algèbre, ce qui efface la
compensation. La méthode standard pour bloquer ça est de marquer le
temporaire `t` comme `volatile`, ce qui force le store / reload depuis
la mémoire et empêche la réassociation au niveau IR :

```cpp
const Real y = cell_at(x, k) - cell_at(cmp, k);
volatile Real t = cell_at(sum, k) + y;
cell_at(cmp, k) = (static_cast<Real>(t) - cell_at(sum, k)) - y;
cell_at(sum, k) = static_cast<Real>(t);
```

### Templating sur `Cell`

Le Kahan add a besoin d'opérer scalaire par scalaire (sinon les
expression templates d'Eigen fusionnent et défont la compensation).
On itère donc sur les composantes via `cell_at<Cell>(c, k)`, qui est
fourni par `CellTraits<Cell>::component` :

- Pour `Cell = Real` : `cell_size = 1`, `cell_at(c, 0) = c`. La boucle
  s'unroll en un seul add.
- Pour `Cell = Eigen::Matrix<Real, N, 1>` : `cell_size = N`,
  `cell_at(c, k) = c[k]`. La boucle s'unroll en N adds à la compile.

Le même header sert donc `FluxRegister<Real>` (advection scalaire) et
`FluxRegister<ConservedState2D>` (Euler 2D), avec la même garantie de
stabilité bit-à-bit.

### Variantes weighted (SSPRK3) et per-thread

`add_F_*_fine_w(k, F, w)` accumule `w * F` au lieu de `F`. Utilisé par
les intégrateurs multi-stage (SSPRK3 Shu-Osher : poids 1/6, 1/6, 4/6
sur les 3 sous-étages). Caller fait `reset()` avant le premier
sous-étage.

`reserve_threads(n)` + `add_F_*_fine_thread(k, F, tid)` +
`merge_thread_accumulators()` : chaque thread accumule dans son slice
`[tid][k]` sans `PDE_OMP_CRITICAL`. À la fin de la région parallèle,
un merge séquentiel Kahan-compensé pousse vers les accumulateurs
canoniques. Sur un M2 8 threads, ça transforme un speedup de 2.7×
(version avec critical) en 3.4× (version per-thread + merge).

## 7. Ghost-fill multi-patch (`ghost_fill2d`)

`fill_patch_ghosts_multipatch<Cell>` remplit les ghost cells d'un
patch `(L, k)` selon deux sources :

```
pour chaque ghost cell (i_local, j_local) du patch :
    1. par défaut : prendre le parent (niveau L-1) à la cellule qui
       contient la position fine du ghost, avec interpolation
       linéaire en temps :
         ghost = (1 - alpha) * parent_n + alpha * parent_np1
       où parent_n est le snapshot au début du coarse step et
       parent_np1 le post-step.
    2. override : si la cellule ghost tombe dans l'INTÉRIEUR d'un
       autre patch de niveau L (sibling), écraser par la valeur
       sibling (pas d'interpolation temporelle, ils sont au même
       time level, donc strictement plus précise que l'injection
       parente).
```

L'invariant clé : avant ce remplissage, le caller doit avoir appliqué
les BC au parent (donc `parent_n` a des ghost valides). C'est ce qui
fait qu'un patch fin qui touche la frontière du domaine global
récupère via le parent des ghosts qui respectent la BC.

Deux passes : d'abord les colonnes ghost gauche / droite sur toute la
hauteur (y compris coins, écrits exactement une fois), puis les
rangées ghost bas / haut sur les colonnes intérieures uniquement.
La boucle interne sur `j` ou `i` est parallèlisable via OpenMP
puisque chaque itération écrit dans une cellule différente.

## 8. Regrid (`regrid{1d,2d}`)

### Single-block `regrid_2d`

Le plus simple : un seul bloc fin par niveau, bbox des cellules
marquées par le critère. Pipeline :

```
marks = criterion(coarse, dx, dy)
(i_lo, i_hi, j_lo, j_hi) = bbox(marks)
expand by `buffer` cells, clamp to domain, round to ratio-2 alignment
if (i_lo, i_hi, j_lo, j_hi) == previous: return false   // pas de changement
snapshot old fine block
mesh.refine_region(i_lo, i_hi, j_lo, j_hi)
fill new fine block:
    for each cell:
        if cell falls inside the old fine box: copy from old (préserve fine data)
        else: piecewise-constant injection from coarse parent
```

L'injection piecewise-constant préserve la masse discrète exactement,
parce que la moyenne des `r` sous-cellules fines égale la valeur coarse
parente (invariant maintenu par l'average-down step à chaque time step).

### Multi-niveau `regrid_2d_multilevel`

Construit la hiérarchie bottom-up jusqu'à `max_target_level` :

```
mesh.clear_refinement()
for L = 1 .. max_target_level:
    marks = criterion(mesh.block(L-1).field, dx_p, dy_p)
    bbox restricted to interior + nest_buffer margin
    if no cells marked: break
    expand, snap to ratio, clamp
    parent_copy = mesh.block(L-1).field()       // avant refine_at_level
    mesh.refine_at_level(L, i_lo, i_hi, j_lo, j_hi)
    fill new level by PCI from parent_copy
```

Le `nest_buffer` est la **proper nesting** de Berger-Colella : un
patch de niveau `L+1` doit avoir au moins `nest_buffer` cellules de
marge depuis le bord de son parent `L`, sinon le ghost-fill `(L+1, L)`
tomberait hors du parent. Défaut : 2 cellules.

### Multi-patch `regrid_2d_multipatch`

Au lieu d'un seul bloc bounding-box, on appelle Berger-Rigoutsos sur
les marks pour obtenir une liste de patches efficaces :

```
marks = criterion(coarse, dx, dy)
boxes = cluster_berger_rigoutsos(marks)
covs = [expand+snap+clamp(b) for b in boxes]
greedy merge covs that now overlap after buffering
if covs == previous patches: return false
mesh.set_patches_at_level(1, covs)
for each patch: PCI from coarse
```

Le greedy merge est O(N^2) sur le nombre de patches, mais N < 10 en
pratique (Berger-Rigoutsos s'arrête vite quand l'efficacité passe
le seuil), donc coût négligeable.

## 9. Macros OpenMP (`PDE_OMP_PARALLEL_FOR`, `PDE_OMP_CRITICAL`)

L'objectif est que le même code source compile sans `#ifdef` interne,
qu'OpenMP soit présent ou non :

```cpp
PDE_OMP_PARALLEL_FOR
for (Index j = 0; j < ny; ++j)
  ...

PDE_OMP_CRITICAL
{ shared_accumulator += local; }
```

Quand `PDE_CORE_HAS_OPENMP` est défini par le build, les macros
expansent en `_Pragma("omp parallel for")` / `_Pragma("omp critical")`.
Sans, elles sont vides. Les solveurs consommateurs (advection_cpp,
euler_cpp) propagent leur propre flag (`ADVECTION_USE_OPENMP` ou
`EULER_USE_OPENMP`) sur ce define pour que les pragmas s'activent en
même temps.

Les helpers `pde_core::omp_num_threads()` et `omp_thread_id()` retombent
sur `1` et `0` respectivement quand OpenMP est absent, ce qui permet
aux per-thread accumulators (`std::vector<T> per_thread(omp_num_threads())`)
de compiler et tourner dans les deux configurations sans branchement.
