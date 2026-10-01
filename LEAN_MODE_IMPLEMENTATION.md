# FastCW lean mode: implementation instructions

## Goal

Make three changes to FastCW, validated and decided with the maintainer:

1. **Lean geometry.** Replace the per-source clipping + bisection loop in `compute_all_wiring_costs` with closed-form geometry (`lean_geometry.py`). Radii become exact roots instead of 1%-area-tolerance bisection results; per-source geometry cost drops ~9x (measured: 95 → 10 ms/source native, 43 → 5 ms/source at 80k vertices).
2. **Robust-only distances with a fixed diffusion length.** potpourri3d always runs `use_robust=True`. `use_robust=False` is no longer an option. The heat-method time step is set as a physical diffusion length `sqrt(t)` in millimetres, default **0.7 mm**, converted per mesh to potpourri3d's `t_coef`.
3. **Safety checks.** A per-source distance-field health check, a label/mesh mismatch guard, and removal of the module-wide warning filter that hid all warnings.

**No batching.** Sources are solved one at a time. Do not port any batched engine; remove the existing `batch_heat` engine.

### Why (for context; do not re-litigate)

- `use_robust=False` corrupts a large share of distance fields on cortical meshes. Measured against exact geodesics (pygeodesic): 57% of fields grossly corrupted on a native surface, MSD off by 25–40%. The cause is non-Delaunay triangles breaking the cotan Laplacian's maximum principle. Robust (intrinsic Delaunay) mode had 0 corrupted fields in every test.
- Heat-method bias depends on the physical diffusion length, not on `t_coef`. Fixing it in mm gives the same bias on every mesh and subject. At 0.7 mm (robust, 80k meshes, two subjects), measured against native exact geodesics:
  - radius error: −3% at the smallest scale to −0.8% at the largest;
  - perimeter error: −1% to −4%;
  - area-weighted MSD bias: about +1%.

  Below ~0.5 mm, fields break down: heat underflows across a hemisphere.

## Files provided with these instructions

| File | Role |
|---|---|
| `lean_geometry.py` | New module. Add to the repo as-is, next to `core_analysis.py`. Numba kernels plus `field_health()` and `mean_edge_length()`. |
| `verify_lean_integration.py` | Acceptance test. Add to the repo. Must pass (see Acceptance). |
| `lean_testkit.py` | Synthetic meshes and stub engines used by the acceptance test. Probably already in the repo; keep it. |
| `reference_patch.diff` | A tested implementation of every change below, as a unified diff against the maintainer's earlier uploaded copies of `core_analysis.py`, `distance_engines.py`, `fastcw.py`, `io_utils.py` and `run_pipeline.py`. |

**Use the diff as the reference, but apply changes by hand against the current repo.** The maintainer has since edited `distance_engines.py` (`PotpourriDistanceEngine.__init__` already has a partial `diffusion_length_mm` block). Other files may also have drifted. Where the diff does not apply cleanly, implement the behaviour described below. The spec in this document is authoritative; the diff shows one correct way to do it.

## Changes, file by file

### 1. `lean_geometry.py` (new)

Add unchanged. Do not modify the kernels: they are validated against the legacy clipping code to ~1e-14 (area) and ~1e-15 (perimeter).

### 2. `core_analysis.py`

1. **Delete** `warnings.filterwarnings('ignore')` near the top. It silences every warning in the process, including the health-check warning added below. Add `import lean_geometry as lg` and a module constant `GEOMETRY_METHOD = "lean_closed_form_v1"`.
2. **Interior non-manifold handling in `__init__`:** remove the branch that switches potpourri to `use_robust=True` (robust is now always on). Print a one-line note for potpourri instead. Keep the existing warning for other engines.
3. **Lean state in `__init__`,** after face geometry and vertex areas are computed:
   - `self._lean_area, self._lean_g11, self._lean_g12, self._lean_g22 = lg.precompute_face_metric(self.vertices, self.faces)`
   - `self._lean_width = lg.default_bin_width(self.vertices, self.faces)`
   - scratch buffers `self._lean_dmax` (float64), `self._lean_order` (int32), `self._lean_long` (int32), each of length `n_faces`, allocated once and reused for every source;
   - `self._health_h = float(np.median(self.face_L))`;
   - `self.field_flag = np.zeros(n_vertices_full, uint8)` and `self.n_flagged_fields = 0`.
4. **Delete** `_compute_geodesic_distance_batch_from_subvertices`.
5. **Add** a static method `_clean_vertex_samples`, moved out of the old loop, with unchanged logic.
6. **Rewrite `compute_all_wiring_costs`.** Keep the signature compatible: `scale, area_tol=None, vertex_subset, n_samples_between_scales=3, boundary_cap_fraction, batch_size=None, verbose`.
   - `area_tol`: ignored. Print a note if it is not None.
   - `batch_size`: ignored. Emit a `DeprecationWarning` if it is not None or 1.
   - Keep the validation of `n_samples_between_scales` and `boundary_cap_fraction`, the result-array resets, the target-area printout, and the vertex-subset handling.
   - Iterate sources one at a time in `self._bfs_order`. Neighbour warm-start logic, `r_sub_by_scale`, `r_euclid` and the dmin/dmax/argsort phase are no longer needed; remove them from the loop.
   - For each source:
     1. `d = self._compute_geodesic_distances_from_subvertex(sub_idx)`.
     2. `neg, viol, bad = lg.field_health(d, self.vertices, sub_idx, self._health_h)`. If `bad`: set `field_flag[orig]=1`, record `(orig, neg, viol)`, leave **every** metric for that vertex NaN (MSD, radii, perimeters, dist_to_boundary), store no samples, and continue.
     3. Distance to boundary and MSD (unweighted and area-weighted): unchanged formulas.
     4. Call `lg.per_source_geometry(d, faces, lean_area, g11, g12, g22, lean_width, targets, extra_fracs, 1e-10, 60, lean_dmax, lean_order, lean_long, radii_out, perim_out, extra_r, extra_a)`, where:
        - `targets` = sorted scales × total cortical area;
        - `extra_fracs = arange(1, m+1)/(m+1)`, with `m = n_samples_between_scales`;
        - `radii_out` and `perim_out` have length `n_scales`;
        - `extra_r` and `extra_a` have length `(n_scales−1)·m`.

        Write `radii_out` and `perim_out` into `radius_function` / `perimeter_function` as float32. NaN is returned when a target exceeds the reachable area; keep it NaN.
     5. Samples:
        - Always store the solved pairs `(radius_s, target_s)` for finite radii.
        - Add the finite supplementary pairs `(extra_r, extra_a)`, dropping any whose `boundary_area_loss_fraction(r, dist_to_boundary)` exceeds `boundary_cap_fraction` when a cap is set.
        - Clean with `_clean_vertex_samples`.
        - The legacy bisection history is no longer produced.
   - After the loop:
     - Build the CSR sample arrays exactly as before.
     - Set `self.n_flagged_fields`.
     - If any sources were flagged, emit `warnings.warn(..., RuntimeWarning)` **and** print a summary with the first 10 flagged vertices.
     - Print a timing summary with four parts (geodesic / MSD+boundary / lean geometry / samples), plus mean band evaluations per vertex.
     - Print the existing MSD, radius and perimeter statistics.
   - Return value unchanged.
7. **Add `provenance()`,** returning a dict with:
   - `geometry_method` and `engine`;
   - `use_robust`, `diffusion_length_mm`, `t_coef` and `mean_edge_length_mm` (read from the engine);
   - `n_flagged_fields`;
   - the two health tolerances (both 1e-3).
8. **Keep, do not delete,** the legacy geometry code and state:
   - the methods `_area_inside_radius`, `_perimeter_at_radius`, `_find_radius_for_area`, `_area_inside_radius_vectorized`;
   - their Numba kernels;
   - the `_f0/_f1/_f2`, `_d*_buf` and `_dmin_buf/_dmax_buf` buffers.

   The acceptance test and the maintainer's diagnostic scripts (`compare_fastcw_lean.py`, `diagnose_mismatch.py`, `tune_heat.py`, `crossmesh_check.py`) call them. Mark them "legacy, validation only" in the module docstring.

### 3. `distance_engines.py`

1. Add module constants `DEFAULT_DIFFUSION_LENGTH_MM = 0.7`, `MIN_DIFFUSION_LENGTH_MM = 0.5` and `RECOMMENDED_MIN_DIFFUSION_LENGTH_MM = 0.7`, plus a `_mean_edge_length(vertices, faces)` helper (mean over unique edges).
2. `PotpourriDistanceEngine.__init__`, required behaviour, in this order:
   1. Pop `allow_eigen_fallback` (existing).
   2. If `use_robust` is in the kwargs: pop it; if it is falsy, raise `ValueError` ("use_robust=False is not supported ..."). True is accepted silently.
   3. If `t_coef` is in the kwargs, raise `ValueError` telling the user to set `diffusion_length_mm`.
   4. `L = float(pop("diffusion_length_mm", 0.7))`:
      - raise `ValueError` if L is not finite or is below 0.5;
      - `warnings.warn(..., RuntimeWarning)` if it is below 0.7.
   5. Import potpourri3d and run the SuiteSparse check (existing code, unchanged).
   6. Set `self.use_robust = True`, `self.diffusion_length_mm = L`, `self.mean_edge_length = _mean_edge_length(V, F)` and `self.t_coef = (L / mean_edge_length)**2`.
   7. Construct `pp3d.MeshHeatMethodDistanceSolver(V, F, **kwargs)`, where kwargs are any remaining engine kwargs with `use_robust=True` and `t_coef=self.t_coef` forced last.

   This replaces the maintainer's partial edit. The conversion must use the mean over unique edges, because that is what the tuning scripts used, so what was validated is what runs.
3. Remove `BatchHeatDistanceEngine`, its `ENGINE_REGISTRY` entry, `BaseDistanceEngine.compute_distance_batch`, `supports_batching`, and the now-unused `scipy.sparse` import.
4. Other engines (`potpourri_fmm`, `pycortex`, `pygeodesic`) are unchanged.

### 4. `fastcw.py`

1. Remove `batch_heat` from the engine choices, and remove its warning block.
2. Add `--diffusion-length-mm` (float, default 0.7). For `--engine potpourri`, set `engine_kwargs["diffusion_length_mm"]` from it:
   - an explicit `--engine-kw diffusion_length_mm=...` wins over the default;
   - raise if both are given explicitly and they differ.

   Do not pass it to other engines.
3. Keep `--use-robust`, `--batch-size` and `--area-tol` as **hidden, deprecated no-ops** (`help=argparse.SUPPRESS`, default None/False), so old command lines still run. When one is given, print a one-line note that it is ignored. Remove the code that set `engine_kwargs["use_robust"]`, and the `batch_size <= 0` validation.
4. Change function-signature defaults to match the CLI: `n_samples_between_scales=3` (was 10), `batch_size=None` and `area_tol=None`.

### 5. `run_pipeline.py`

1. Add `--diffusion-length-mm` (default 0.7), always forwarded to fastcw.
2. Keep `--use-robust`, `--batch-size` and `--area-tol` as hidden no-ops, and **do not forward them**. (`--area-tol` is currently always forwarded; stop that.)

### 6. `io_utils.py`

1. `_read_freesurfer_label_mask`: if any label index is outside `[0, n_vertices)`, raise `ValueError`. The message must name the label file and the maximum index, and suggest `--no-mask` for cortex-only surfaces. Out-of-range indices were previously dropped silently, which scrambled masks on decimated surfaces. On native surfaces all indices are in range, so nothing changes there.
2. `load_surface_and_mask` (FreeSurfer branch): if the surface file name contains `.cortexonly.` and `no_mask` is False:
   - raise `ValueError` if `mask_path` or `custom_label` was given explicitly;
   - otherwise print a note, treat it as `no_mask=True`, and set `mask_source="cortexonly_surface_all_vertices"`.
3. `save_analysis_npz`:
   - add `field_health_flag` (uint8 per vertex) when the analysis has `field_flag`;
   - add `provenance_json` (the `json.dumps` of `analysis.provenance()`, sorted keys) when it has `provenance`;
   - in that case bump `fastcw_output_schema_version` to 3.
   - The CSV, MGH and GIFTI outputs are unchanged.

## Acceptance

All of these must pass before the work is considered done.

1. **Synthetic acceptance test** (no data or potpourri3d needed):

   ```bash
   python verify_lean_integration.py --synthetic 5
   ```

   It must print `ALL CHECKS PASSED` (17 checks) and exit with status 0. It checks:
   - integrated radii reproduce target areas under the legacy clipping code;
   - perimeters, samples and MSD match the legacy code;
   - a corrupted field is flagged, set to NaN and warned about;
   - the engine forces robust mode with the correct `t_coef`;
   - `use_robust=False`, `t_coef` and a 0.4 mm diffusion length are rejected, and 0.6 mm warns;
   - `batch_heat` is gone;
   - no blanket warning filter remains.
2. **Real-subject acceptance test** on a reduced mesh and on native:

   ```bash
   S=/data/users/jlee38/homefolders/Documents/FastCorticalWiring/testsub
   python verify_lean_integration.py $S sub-100604200_ses-1 --hemi lh --surf-type pial.cortexonly.qd.n80000 --n-sources 20
   python verify_lean_integration.py $S sub-100604200_ses-1 --hemi lh --n-sources 20
   ```

   Both must pass, and "no fields flagged on this run" must pass. With robust mode at 0.7 mm, no flags are expected on these subjects. A flag here means something is wrong: stop and report it; do not loosen the check.
3. **End-to-end smoke run** on one subject with a vertex subset. Confirm:
   - outputs are written;
   - the NPZ contains `field_health_flag` and `provenance_json`, and the latter shows `use_robust: true`, `diffusion_length_mm: 0.7` and `geometry_method: lean_closed_form_v1`;
   - the log shows the new timing breakdown.

   ```bash
   python run_pipeline.py subjects.txt --subjects-dir $S --output-label lean_v4_test --surf-type pial.cortexonly.qd.n80000 --vertex-list some_vertices.txt -j 1
   ```
4. **Old command lines still run.** Passing `--use-robust`, `--batch-size 32` or `--area-tol 0.01` to `fastcw.py` or `run_pipeline.py` prints a deprecation note and runs normally.
5. **Forbidden settings fail loudly.** `--engine-kw use_robust=false` and `--engine-kw t_coef=1` each fail with a clear error.

Do **not** use agreement with outputs from before this change as a pass criterion. Previous outputs used non-robust distances and differ substantially by design.

## Do not

- Do not modify the kernels in `lean_geometry.py`, or the health-check thresholds (1e-3 for both fractions, with h = median face max-edge).
- Do not add any batched solver, or a per-scale or per-metric second solve.
- Do not delete the legacy geometry methods, their kernels, or the `_d*_buf` buffers (see step 2.8).
- Do not change the default scale set. The class default (`FastCorticalWiringAnalysis.DEFAULT_SCALES`, 6 scales from 0.0006 to 0.016) differs from `run_pipeline.py` and the README (12 scales from 0.002 to 0.05). That is a known open decision for the maintainer; list it in your summary rather than choosing.
- Do not re-enable `use_robust=False` under any flag name.

## Behaviour changes to note in the changelog / README

- Distances: potpourri3d robust mode only, with the diffusion length fixed at 0.7 mm by default (`--diffusion-length-mm`). Outputs are not comparable with earlier versions; re-run existing results.
- Radii are exact (no `--area-tol`). The bisection history is no longer stored in the samples NPZ. Each vertex stores 12 solved pairs plus up to 11 × `n_samples_between_scales` supplementary pairs.
- Sources with corrupted distance fields are written as NaN and listed in `field_health_flag`.
- New NPZ fields: `field_health_flag`, `provenance_json`; schema version 3.
- `batch_heat` engine removed; `--use-robust`, `--batch-size` and `--area-tol` are deprecated no-ops.
- Cortex-only surfaces (`*.cortexonly.*`) automatically use all vertices, and refuse explicit masks. Labels that do not match the mesh now raise an error instead of being silently truncated.
- README corrections needed:
  - "fastcw uses robust heat method" is now true (it was not before);
  - remove `batch_heat` from the backend list;
  - "performance scales approximately linearly with number of vertices" is true per vertex but roughly quadratic per hemisphere;
  - the performance table mentions "Anisotropy", which is not computed anywhere.

## Out of scope (separate changes, do not include)

- Replacing `preexec_fn` CPU pinning in `run_pipeline.py` with a `numactl`/`taskset` command prefix. `preexec_fn` is documented as unsafe with threads.
- Balancing workers per L3 cache domain rather than per socket.
- The default-scale mismatch described above.
