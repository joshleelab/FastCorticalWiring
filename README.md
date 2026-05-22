---

# Fast Cortical Wiring

High-performance Python tools for computing intrinsic cortical wiring metrics from surface meshes, following the framework of Ecker et al. (2013). This implementation is optimized for modern cortical mesh resolutions (fsaverage5–fsaverage and higher) and uses state-of-the-art geodesic solvers to achieve practical runtimes on large surfaces.

---

## Important: SuiteSparse/CHOLMOD Build Prerequisites

`potpourri3d` performs best when built against SuiteSparse, and the `batch_heat` engine requires CHOLMOD through `scikit-sparse`. Install build prerequisites before installing Python dependencies:

- macOS (Homebrew): `brew install suitesparse`
- Ubuntu/Debian: `apt-get install cmake libsuitesparse-dev`
- Windows: SuiteSparse setup is often difficult; use WSL/Linux when possible.

After SuiteSparse/CMake are installed, install `potpourri3d` from source:

```bash
pip install -vvv --no-cache-dir --no-binary potpourri3d potpourri3d
```

This source build behavior is also enforced in `requirements.txt`.

To verify the installed wheel is actually linked against SuiteSparse:

```bash
python backend_check.py
```

Expected result: `PASS: SuiteSparse linkage detected.`

---

## Overview

The cerebral cortex is a folded 2-dimensional sheet embedded in 3D. Many biologically meaningful distance measures—such as connection length or wiring cost—must be computed along the surface itself, not through Euclidean space.

This package computes intrinsic wiring metrics at each cortical vertex:

* **Mean Separation Distance (MSD)**
  Emitted as `msd_unweighted` (Ecker 2013 arithmetic mean over vertices) and
  `msd_weighted` (area-weighted surface-integral approximation).

* **Radius Function (Local Wiring Scale)**
  The geodesic radius required to enclose a fixed fraction of cortical surface area (typically 5%).

* **Perimeter Function (Boundary Wiring Cost)**
  The perimeter of the geodesic disc defined by the radius function. Related to the cost of connecting to neighboring cortical territories.

* **Distance to Boundary**
  Geodesic distance from each vertex to the nearest physical submesh boundary, useful for filtering large-radius finite-size effects.

* **Sampled Radius/Area Pairs**
  Bisection and supplementary `(radius, area)` samples are saved so downstream scale-law fitting can be redone without rerunning geodesics.

These metrics provide quantitative descriptions of cortical spatial organization and intrinsic wiring constraints.

---

## Key Features

### High-performance geodesic computation

Uses pluggable geodesic backends:

* `potpourri3d` (primary backend, recommended)
* `batch_heat` (batched heat-method backend using robust-laplacian + CHOLMOD)
* `pygeodesic` (exact discrete-geodesic reference backend)
* `pycortex` (alternate backend)

For `potpourri3d` and `pycortex`, the heat method solves geodesic distance to all vertices simultaneously using sparse linear algebra. This is dramatically faster than graph-based methods like Dijkstra’s algorithm for the one-to-all distance computations required here.

Typical speedup over traditional approaches: **10×–100×**

---

### Optimized geometric computations

Performance-critical operations are optimized using several techniques:

* **Numba JIT compilation**
  Accelerates geometric clipping, area integration, and perimeter extraction loops.

* **Precomputed mesh geometry**

  * Face areas
  * Vertex areas
  * Edge lengths
  * Median mesh length scale (L)

* **BFS-ordered vertex processing**

  * Improves locality
  * Enables warm-started radius bracketing
  * Reduces iterations needed for radius inversion

* **Robust radius inversion**

  * Uses bracketed root-finding for stability
  * Avoids Newton-method instability on discretized meshes

---

### Designed for real cortical mesh scales

Fully validated at realistic resolutions:

| Mesh       | Vertices / hemi |
| ---------- | --------------- |
| fsaverage5 | ~10k            |
| fsaverage6 | ~41k            |
| fsaverage  | ~164k           |

Validation uses analytic surfaces with known ground truth and area-fraction-based radius targets.

---

### Cortex-only submesh support

Automatically restricts computation to cortex label:

* excludes medial wall
* reduces computational load
* matches typical neuroimaging workflows
* hard-fails if no cortex mask can be resolved (unless `--no-mask` is explicitly set)

---

### Standard neuroimaging outputs

Produces both analysis-ready and visualization-ready outputs:

* CSV tables with per-vertex metrics (`msd_unweighted`, `msd_weighted`, and per-scale local measures)
* FreeSurfer `.mgh` overlays
* fsLR `.shape.gii` overlays
* compressed `*_wiring_samples.npz` files with sampled `(radius, area)` pairs
* compatible with FreeSurfer, nilearn, PySurfer, etc.

---

## Repository contents

```
core_analysis.py
    Shared cortical wiring pipeline (engine-agnostic)

distance_engines.py
    Backend adapters (potpourri, pygeodesic, pycortex)

fastcw.py
    Canonical CLI interface and orchestration (default engine: potpourri)

run_pipeline.py
    NUMA-aware batch runner for native-space subject processing

one_off_scripts/fastcw_derived_metrics_to_schaefer.py
    Native-space parcel summary helper for radius/perimeter-derived metrics
```

---

## Installation

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Required dependencies:

```
numpy
scipy
nibabel
potpourri3d
numba
matplotlib
tqdm
pycortex
pygeodesic
robust-laplacian
scikit-sparse<0.5
```

For best performance, build potpourri3d locally with SuiteSparse. The `batch_heat` backend also needs SuiteSparse/CHOLMOD headers available when installing `scikit-sparse<0.5`; the 0.5+ API requires newer SuiteSparse than many system packages provide.

---

## Usage

Primary tool:

```
python fastcw.py <subjects_dir> <subject_id>
```

Common options:

```
--hemispheres lh rh
--surf-type pial
--custom-label cortex
--scale 0.002 0.00267988 ... 0.05
--area-tol 0.01
--engine potpourri|batch_heat|pygeodesic|pycortex
--batch-size 32
--n-samples-between-scales 3
--boundary-cap-fraction 0.05
--verbose-timing
--overwrite
```

MSD is always computed. The default scale set contains 12 log-spaced local area fractions from `0.002` to `0.05`.

Example:

```
python fastcw.py $SUBJECTS_DIR fsaverage \
  --hemispheres lh rh \
  --scale 0.005 0.01 0.05
```

The CLI writes scalar outputs plus a compressed sample archive named like:

```
{subject}_{hemi}_{surf}_wiring_samples.npz
```

The NPZ contains:

* `sample_radii_flat`, `sample_areas_flat`: flat CSR-style `float32` sample arrays
* `sample_indptr`: `int64` row offsets; vertex `v` samples are `flat[indptr[v]:indptr[v+1]]`
* `sample_scales_solved`, `n_samples_between_scales`, `boundary_cap_fraction`
* `cortex_mask`, `sub_to_orig`

---

## Validation

Synthetic validation using analytic surfaces:

```
python validate_fastcw_pp3d.py \
  --base-dir test_surfaces_fs_like \
  --regenerate
```

Optional full integration validation:

```
python validate_fastcw_pp3d.py \
  --base-dir test_surfaces_fs_like \
  --run-integration
```

Validation includes:

* spheres
* planes
* cylinders
* fsaverage-like resolutions (fs5/fs6/fs7)
* area fractions `{0.01, 0.03, 0.05, 0.10}` with explicit 5% reporting
* mesh-size and mesh-scale gates (`nV >= 10,000`, `r >= 5 * median_edge_length`)
* per-case interior/local-disc filtering (boundary-intersecting cases are skipped and reported)

Tests explicitly verify accuracy at realistic 5% surface area patches.

---

## Performance

Runtime depends primarily on vertex count and number of requested scales.

Typical performance on fsaverage6 (~41k vertices):

| Metric                          | Time                          |
| ------------------------------- | ----------------------------- |
| Radius + Perimeter + Anisotropy | seconds–minutes              |
| MSD                             | minutes                       |
| Full hemisphere                 | practical on workstation CPUs |

Performance scales approximately linearly with number of vertices.

The radius phase uses vectorized fully-inside triangle accumulation and band-only clipping around isolines. `--n-samples-between-scales` adds a small amount of area-integration work after the scale radii have been solved. `--boundary-cap-fraction` filters only supplementary samples by estimated boundary-clipped disc area loss; converged target-scale samples are always retained. The value is an area-loss fraction under a locally Euclidean, locally planar-boundary circular-segment approximation: `0.0` rejects once a supplementary disc crosses the boundary, `0.05` permits up to 5% loss, and values must be in `[0, 0.5)`. Leave it unset or use `--boundary-cap-fraction none` for no cap. This is not comparable to the old geometric radius-fraction semantics; old values should be reselected explicitly.

---

## Algorithmic background: the heat method

Traditional shortest-path algorithms traverse mesh edges sequentially. These methods become inefficient when computing distances from every vertex.

The heat method instead solves a pair of sparse linear systems:

Step 1: simulate heat diffusion from a source vertex
Step 2: compute the normalized temperature gradient field
Step 3: integrate the gradient field to recover geodesic distance

This produces accurate geodesic distance fields efficiently using optimized sparse solvers.

Advantages:

* robust to mesh irregularities
* highly parallelizable
* efficient for repeated distance solves
* ideal for cortical surface analysis

Reference:

Crane et al., Geodesics in Heat (SIGGRAPH 2013)

---

## Scientific interpretation

These metrics describe intrinsic wiring geometry of cortex.

### Mean Separation Distance

Represents global wiring cost.

Higher MSD:

* spatially isolated regions
* association cortex

Lower MSD:

* spatially central regions
* sensory cortex

---

### Radius function

Represents spatial scale of local cortical neighborhoods.

Smaller radius:

* denser cortical packing
* shorter intrinsic wiring cost

Larger radius:

* expanded cortical areas
* longer intrinsic wiring distances

---

### Perimeter function

Represents boundary extent between cortical regions.

Related to inter-area connection costs and cortical topology.

---

## Optimization summary

Major performance contributors:

| Optimization                   | Impact                        |
| ------------------------------ | ----------------------------- |
| Heat method geodesics          | dominant speed improvement    |
| Sparse linear algebra          | scalable distance solves      |
| Numba-compiled geometry        | 10×–100× speedup              |
| Submesh restriction            | reduces problem size          |
| Warm-started radius bracketing | fewer root-finding iterations |
| Precomputed geometry           | avoids redundant computation  |

---

## Current backend recommendation

Use:

```
fastcw.py
```

This uses potpourri3d with robust heat method solver and is actively maintained.

The default backend is potpourri3d; pycortex and pygeodesic are alternate engines.

---

## References

Crane et al. (2013)
Geodesics in Heat

Ecker et al. (2013)
Intrinsic wiring cost in the cerebral cortex

Gao et al. (2015)
pycortex: interactive surface visualization

Sharp et al. (2019–)
potpourri3d geometry processing library

---

## Summary

This package provides a fast, scalable, and validated implementation of intrinsic cortical wiring metrics suitable for modern cortical surface datasets.
